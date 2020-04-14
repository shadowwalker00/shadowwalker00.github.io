---
title: Linux命令总结
date: 2019-03-04 12:03:12
tags:	
---


## 解压
解压 tar.gz
```shell
tar -zxvf ×××.tar.gz
```

## 打包
```shell
tar -zcvf [package].tar.gz /[dirname]
```
<!--more-->
## Bind the portable hard drive
- 运行  ```sudo nano /etc/fstab```
- 写入这个文件 LABEL=NAME none ntfs rw,auto,nobrowse
- 打开 /Volumns






