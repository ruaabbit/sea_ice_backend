import asyncio
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import aiohttp
from dateutil import relativedelta

# 常量定义
BASE_URL_PRE_2021 = "https://thredds.met.no/thredds/fileServer/osisaf/met.no/reprocessed/ice/conc_450a_files"
BASE_URL_POST_2021 = "https://thredds.met.no/thredds/fileServer/osisaf/met.no/reprocessed/ice/conc_cra_files"
BASE_URL_MONTHLY = "https://thredds.met.no/thredds/fileServer/osisaf/met.no/reprocessed/ice/conc_cra_files/monthly"

PRIORITY = {"cdr": 1, "icdr": 2, "icdrft": 3}

FILENAME_PATTERN_DAILY = (
    r"^ice_conc_nh_ease2-250_(cdr|icdr|icdrft)-v3p0_(\d{4})(\d{2})\d{2}1200\.nc$"
)
FILENAME_PATTERN_MONTHLY = (
    r"^ice_conc_nh_ease2-250_(cdr|icdr|icdrft)-v3p0_(\d{4})(\d{2})\.nc$"
)

FILENAME_PATTERN_ALL = (
    r"^ice_conc_nh_ease2-250_(cdr|icdr|icdrft)-v3p0_(\d{4}\d{2}(?:\d{6})?)\.nc$"
)


async def download_file(session, file_url, output_file, retries=3, delay=3):
    """
    异步下载文件，支持重试机制。
    """
    if output_file.exists():
        print(f"文件已存在，跳过下载: {output_file}")
        return True

    for attempt in range(retries):
        try:
            async with session.get(file_url) as response:
                if response.status != 200:
                    raise aiohttp.ClientError(f"HTTP 状态码 {response.status}")
                data = await response.read()
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with output_file.open("wb") as f:
                    f.write(data)
                # print(f"下载成功: {output_file}")
                return True
        except aiohttp.ClientError as e:
            return False
        except Exception as e:
            # print(f"第 {attempt} 次尝试下载失败 {file_url}: {e}")
            await asyncio.sleep(delay)
    # print(f"下载失败: {file_url} 在 {retries} 次尝试后")
    return False


def organize_file(file_path, base_directory):
    """
    Organize files into corresponding year and month directories based on the filename.
    """
    file_path = Path(file_path)
    base_directory = Path(base_directory)
    filename = file_path.name

    # Try to match the filename with daily pattern
    match = re.match(FILENAME_PATTERN_DAILY, filename)
    if match:
        year, month = match.group(2), match.group(3)
        target_directory = base_directory / year / month
    else:
        # Try to match the filename with monthly pattern
        match = re.match(FILENAME_PATTERN_MONTHLY, filename)
        if match:
            year = match.group(2)
            target_directory = base_directory / year
        else:
            print(f"Filename does not match any pattern, cannot organize: {filename}")
            return

    target_directory.mkdir(parents=True, exist_ok=True)
    target_path = target_directory / filename
    shutil.move(str(file_path), str(target_path))
    # print(f"File moved to: {target_path}")
    return target_path


def generate_tasks(start_date, end_date, output_directory, task_type='DAILY'):
    """
    生成需要下载的文件任务列表。
    """
    output_directory = Path(output_directory)
    tasks = []
    current_date = start_date

    if task_type == 'DAILY':
        while current_date <= end_date:
            file_date = current_date.strftime("%Y%m%d")
            if current_date.year < 2021:
                base_url = BASE_URL_PRE_2021
            else:
                base_url = BASE_URL_POST_2021
            for suffix in ['cdr', 'icdr', 'icdrft', ]:
                filename = f"ice_conc_nh_ease2-250_{suffix}-v3p0_{file_date}1200.nc"
                file_url = f"{base_url}/{current_date.year}/{current_date.month:02d}/{filename}"
                output_file = output_directory / filename
                tasks.append((file_url, output_file))
            current_date += relativedelta.relativedelta(days=1)
    else:
        while current_date <= end_date:
            file_date = current_date.strftime("%Y%m")
            for suffix in ['cdr', 'icdr', 'icdrft', ]:
                filename = f"ice_conc_nh_ease2-250_{suffix}-v3p0_{file_date}.nc"
                file_url = f"{BASE_URL_MONTHLY}/{current_date.year}/{filename}"
                output_file = output_directory / filename
                tasks.append((file_url, output_file))

            current_date += relativedelta.relativedelta(months=1)

    return tasks


async def download_and_organize_data(start_date, end_date, output_directory, task_type='DAILY', max_connections=10):
    """
    异步下载并组织数据文件。
    """
    output_directory = Path(output_directory)
    if not output_directory.exists():
        output_directory.mkdir(parents=True)
    tasks = generate_tasks(start_date, end_date, output_directory, task_type)

    connector = aiohttp.TCPConnector(limit=max_connections)
    downloaded_files = []

    async with aiohttp.ClientSession(connector=connector) as session:
        download_tasks = [
            download_file(session, file_url, output_file)
            for file_url, output_file in tasks
        ]
        results = await asyncio.gather(*download_tasks)
        list_download_success = [str(output_file) for (file_url, output_file), success in zip(tasks, results) if
                                 success]

        # 分组并按优先级保留最高的文件
        deduplicated_files_dict = defaultdict(lambda: ('', float("inf")))  # (file_path, priority)
        for file_path in list_download_success:
            match = re.search(FILENAME_PATTERN_ALL, file_path.split("/")[-1])
            if match:
                group1, group2 = match.groups()
                priority = PRIORITY[group1]
                # 如果当前文件优先级更高，则替换
                if priority < deduplicated_files_dict[group2][1]:
                    deduplicated_files_dict[group2] = (file_path, priority)

        # 提取最终结果
        deduplicated_files = [file for file, _ in deduplicated_files_dict.values()]

        for output_file in deduplicated_files:
            try:
                # 只处理非重复的文件或被选中的重复文件
                organized_file = organize_file(output_file, output_directory)
                downloaded_files.append(organized_file)

            except Exception as e:
                print(f"组织文件时出错 {output_file}: {e}")
    return downloaded_files


if __name__ == "__main__":
    start_date = datetime(2023, 12, 30)
    end_date = datetime(2024, 10, 31)
    output_directory = Path(r"C:\Users\qq154\Desktop\test")

    result = asyncio.run(download_and_organize_data(start_date, end_date, output_directory, task_type='MONTHLY'))
    print(result)
