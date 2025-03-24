"""
Author: 爱吃菠萝 1690608011@qq.com
Date: 2024-08-30 11:04:51
LastEditors: 爱吃菠萝 1690608011@qq.com
LastEditTime: 2024-09-08 15:46:59
FilePath: /root/amsr2-asi_daygrid_swath-n6250/data/check.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
"""

from datetime import datetime, timedelta

# 定义起始和结束日期
start_date = datetime.strptime("19790101", "%Y%m%d")
end_date = datetime.strptime("20231231", "%Y%m%d")

# 从文件中读取文件路径
file_paths = []
with open("nc_sic_path.txt", "r") as file:
    for line in file:
        file_paths.append(line.strip())

# 提取日期并放入集合中
existing_dates = set()
for file_path in file_paths:
    date_str = file_path.split("_")[-3]
    existing_dates.add(datetime.strptime(date_str, "%Y%m%d"))

# 遍历日期范围并检查是否有遗漏的文件
missing_dates = []
current_date = start_date
while current_date <= end_date:
    if current_date not in existing_dates:
        missing_dates.append(current_date.strftime("%Y%m%d"))
    current_date += timedelta(days=1)

# 输出遗漏的日期
if missing_dates:
    print("Missing dates:")
    for date in missing_dates:
        print(date)
else:
    print("No missing files.")
