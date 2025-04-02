""" Download and organize sea ice concentration data from NSIDC"""

import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import requests


def download_file(file_url, output_file, retries=3, delay=5):
    for attempt in range(retries):
        try:
            response = requests.get(file_url)
            response.raise_for_status()  # Raise HTTPError for bad responses

            with open(output_file, "wb") as f:
                f.write(response.content)
            print(f"Downloaded: {output_file}")
            return True

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed to download {file_url}: {e}")
            time.sleep(delay)

    print(f"Failed to download {file_url} after {retries} attempts")
    return False


def organize_file(current_directory, filename):
    # 修正后的正则表达式
    pattern = r"^sic_psn25_(\d{4})(\d{2})\d{2}_.+\.nc$"
    match = re.match(pattern, filename)

    if match:
        year, month = match.groups()
        year_directory = os.path.join(current_directory, year)
        month_directory = os.path.join(year_directory, month)

        # 创建年份和月份文件夹
        os.makedirs(month_directory, exist_ok=True)

        # 组织文件路径
        source_path = os.path.join(current_directory, filename)
        target_path = os.path.join(month_directory, filename)

        # 检查文件是否存在，避免异常
        if os.path.exists(source_path):
            shutil.move(source_path, target_path)
            print(f"Moved {filename} to {month_directory}")
        else:
            print(f"File not found: {source_path}")
    else:
        print(f"Filename did not match pattern: {filename}")


def download_and_organize_data(start_date, end_date, output_directory, max_workers=5):
    base_url = "https://noaadata.apps.nsidc.org/NOAA/G02202_V5/north/daily"
    tasks = []

    current_date = start_date
    while current_date <= end_date:
        file_date = current_date.strftime("%Y%m%d")

        if current_date <= datetime(1987, 7, 9):
            filename = f"sic_psn25_{file_date}_n07_v05r00.nc"

        elif current_date <= datetime(1991, 12, 2):
            filename = f"sic_psn25_{file_date}_F08_v05r00.nc"

        elif current_date <= datetime(1995, 9, 30):
            filename = f"sic_psn25_{file_date}_F11_v05r00.nc"

        elif current_date <= datetime(2007, 12, 31):
            filename = f"sic_psn25_{file_date}_F13_v05r00.nc"

        elif current_date <= datetime(2024, 12, 31):
            filename = f"sic_psn25_{file_date}_F17_v05r00.nc"

        file_url = f"{base_url}/{current_date.year}/{filename}"
        output_file = os.path.join(output_directory, filename)

        tasks.append((file_url, output_file, filename))

        current_date += timedelta(days=1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(download_file, file_url, output_file): (
                file_url,
                output_file,
                filename,
            )
            for file_url, output_file, filename in tasks
        }

        for future in as_completed(future_to_task):
            file_url, output_file, filename = future_to_task[future]
            try:
                success = future.result()
                if success:
                    organize_file(output_directory, filename)
            except Exception as e:
                print(f"Failed to process {file_url}: {e}")


if __name__ == "__main__":
    start_date = datetime(1979, 1, 1)
    end_date = datetime(2023, 12, 31)
    output_directory = "/data1/Arctic_Ice_Forecasting_Datasets/Sea_Ice_Concentration"

    os.makedirs(output_directory, exist_ok=True)

    download_and_organize_data(start_date, end_date, output_directory)
