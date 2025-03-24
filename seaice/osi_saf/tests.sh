#!/bin/bash
###
 # @Author: 爱吃菠萝 1690608011@qq.com
 # @Date: 2024-06-25 16:06:28
 # @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 # @LastEditTime: 2025-01-31 21:33:17
 # @FilePath: /Oscar/Cross-modality/test.sh
 # @Description:
 #
 # Copyright (c) 2024 by ${git_name_email}, All Rights Reserved.
###

START_DATE="20231201"
END_DATE="20231231"
GRAD_DAY=0                   # 默认计算第0天的梯度（0-based）
GRAD_TYPE="sum"              # 默认梯度计算方式
SAVE_RESULT=false            # 是否保存结果

INPUT_LENGTH=7
PREDICTION_LENGTH=7

DAYS_TO_ADD=$((INPUT_LENGTH + PREDICTION_LENGTH - 1))

# 将日期字符串转换为自纪元以来的秒数
start_seconds=$(date -d "$START_DATE" +%s)
end_seconds=$(date -d "$END_DATE" +%s)

# 计算调整后的结束日期秒数，减去 DAYS_TO_ADD
adjusted_end_seconds=$(date -d "$END_DATE - $DAYS_TO_ADD days" +%s)

# 在范围内循环遍历每一天
current_seconds=$start_seconds
while [ $current_seconds -le $adjusted_end_seconds ]; do
    current_start_date=$(date -d "@$current_seconds" +%Y%m%d)
    current_end_date=$(date -d "$current_start_date + $DAYS_TO_ADD days" +%Y%m%d)

    # 执行测试命令
    python model_interpreter.py -st $current_start_date -et $current_end_date --grad_day $GRAD_DAY --grad_type $GRAD_TYPE --save

    # 检查上一条命令的退出状态
    if [ $? -ne 0 ]; then
        echo "Error occurred while executing python test.py -st $current_start_date -et $current_end_date"

        exit 1
    fi

    # 移动到下一天
    current_seconds=$(date -d "$current_start_date + 1 day" +%s)
done