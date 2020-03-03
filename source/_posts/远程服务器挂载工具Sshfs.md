---
title: 远程服务器挂载工具Sshfs
date: 2018-02-10 21:13:43
tags: Sshfs
categories: 配置
---

# 背景

在开发过程中，我们很多时候需要远程访问服务器。对于认为Vim十分灵活的大神这个感觉感觉完全没有必要。对于我这种习惯了本地开发的同学，该软件能够帮助你将远程服务器文件夹映射到本地，然后直接在本地进行开发即可。

<!-- more --> 

# Windows版本
##准备文件
我的系统是Win10，首先依次安装这两个软件[DokanSetup.exe](https://github.com/dokan-dev/dokany/releases)和[WinSSHFS-1.6.1.13-devel.msi](https://github.com/Foreveryone-cz/win-sshfs/releases)。
##使用
安装后的软件情况如下：
![这里写图片描述](https://i.loli.net/2018/10/25/5bd0e19f99cd6.png)
安装后分别输入：
服务器部分：

- 驱动的名字
- 主机IP
- 端口Port
- 用户名Username
- 密码Password
- 根目录Directory

本机部分：

- 驱动号Drive Letter

然后单击Save后挂载Mount。
此时，查看我的电脑，便会发现会出现一个新的盘符。
![这里写图片描述](http://img.blog.csdn.net/20180210210650746?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyOTc3NzY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
此时点击进入就是服务器的根目录。
如果像取消挂载，则点击Unmount。
#Mac版本
前面没有提到，首先要确认远程访问的服务器开启了sshfs服务。可通过查看 /etc/ssh/sshd_config 中是否有 Subsystem sftp/usr/lib/openssh/sftp-server确认。
![这里写图片描述](https://i.loli.net/2018/10/25/5bd0e19f9f44b.png)

## 准备文件
接着分别安装osxfuse和sshfs(确保已经安装了homebrew)
```
brew cask install osxfuse
brew install sshfs
```
##使用
```
#挂载：注意本地文件夹名不要带空格
sshfs -p port usr_name@ssh_server_ip:remote_dir local_dir
#取消挂载：注意这时候应该关闭文件夹，或者可以加上-f强制取消
umount local_dir
umount -f local_dir
```
![这里写图片描述](https://i.loli.net/2018/10/25/5bd0e19fd7d5d.png)