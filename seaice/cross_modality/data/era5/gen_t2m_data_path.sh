#!/bin/bash
###
 # @Author: 爱吃菠萝 1690608011@qq.com
 # @Date: 2024-08-17 16:44:45
 # @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 # @LastEditTime: 2025-01-09 22:59:17
 # @FilePath: /Oscar/multimodal-n25/data/era5/gen_t2m_data_path.sh
 # @Description: 生成nc数据路径
 # 
 # Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
### 

DATA_DIR=/data/Arctic_Ice_Forecasting_Datasets/ERA5/t2m
TEXTFILE=/home/ubuntu/sea_ice_backend/seaice/cross_modality/data/t2m_path.txt

for year in `ls $DATA_DIR`
do
    if [ -d "${DATA_DIR}/${year}" ]; then
        for month in `ls ${DATA_DIR}/${year}`
        do
            if [ -d "${DATA_DIR}/${year}/${month}" ]; then
                for datafile in `ls ${DATA_DIR}/${year}/${month}`
                do
                    if [[ "$datafile" == *.npy ]]; then
                        echo ${DATA_DIR}/${year}/${month}/$datafile >> $TEXTFILE
                    fi
                done
            fi
        done
    fi
done