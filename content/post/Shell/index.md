---
title: Shell
date: 2017-09-01
image: 
description: Shell相关技术内容的简单介绍
math: 
categories:
  - 技术简介
tags:
---

类 Unix 系统中的命令行解释器，可以输入命令或编写脚本来操控系统中的文件、设备等

> A **Unix shell** is a command-line [interpreter](https://en.wikipedia.org/wiki/Interpreter_(computing) "Interpreter (computing)") or [shell](https://en.wikipedia.org/wiki/Shell_(computing) "Shell (computing)") that provides a command line [user interface](https://en.wikipedia.org/wiki/User_interface "User interface") for [Unix-like](https://en.wikipedia.org/wiki/Unix-like "Unix-like") [operating systems](https://en.wikipedia.org/wiki/Operating_system "Operating system"). The shell is both an interactive [command language](https://en.wikipedia.org/wiki/Command_language "Command language") and a [scripting language](https://en.wikipedia.org/wiki/Scripting_language "Scripting language"), and is used by the operating system to control the execution of the system using [shell scripts](https://en.wikipedia.org/wiki/Shell_script "Shell script").
> 
 > Users typically interact with a Unix shell using a [terminal emulator](https://en.wikipedia.org/wiki/Terminal_emulator "Terminal emulator"); however, direct operation via serial hardware connections or [Secure Shell](https://en.wikipedia.org/wiki/Secure_Shell "Secure Shell") are common for server systems. All Unix shells provide filename [wildcarding](https://en.wikipedia.org/wiki/Wildcard_character "Wildcard character"), [piping](https://en.wikipedia.org/wiki/Pipeline_(Unix) "Pipeline (Unix)"), [here documents](https://en.wikipedia.org/wiki/Here_document "Here document"), [command substitution](https://en.wikipedia.org/wiki/Command_substitution), [variables](https://en.wikipedia.org/wiki/Variable_(programming) "Variable (programming)") and [control structures](https://en.wikipedia.org/wiki/Control_flow "Control flow") for [condition-testing](https://en.wikipedia.org/wiki/Conditional_(programming) "Conditional (programming)") and [iteration](https://en.wikipedia.org/wiki/Iteration "Iteration").

## Shell 基础

Shell 分为登录式 Shell 和非登录式 Shell：
- 登录式 Shell：
    - 正常通过某终端登录的 shell。
    - `su - username`。
    - `su -l username`。
- 非登录式 Shell：
    - `su username`。
    - 图形终端下打开的命令窗口。
    - 自动执行的 shell 脚本。

在 Shell 环境里可以执行各种命令（交互式），也可以通过运行脚本形式执行预先写好的命令（非交互式）。运行脚本的方式分为几种：
1. `sh file.sh` 或 `bash file.sh` 或 `./file.sh`：都是通过启动子进程在另一个 Shell 中运行命令的方式，最后一种需要有可执行权限。执行完后会销毁该子 Shell，所以环境变量修改不会影响父 Shell
2. `source file.sh` 或 `. file.sh`：在当前 Shell 里执行脚本里的命令。执行时会改变当前 Shell 的环境

Shell 启动或登录后会自动运行一套脚本来初始化环境，修改的方法有：
1. `export XXX=XXX`：修改当前 Shell 的环境变量，作用域仅限于当前 shell 窗口或脚本内
2. `/etc/profile`：用户登录时会第一个执行的脚本，之后可能会读取用户配置 `~/.profile` 或者 `~/.bash_profile` 之类的作后续补充
3. `~/.bashrc`：非登录式 Shell 会加载的脚本，也就是每次图形界面下打开终端或者运行脚本时都会读取的脚本，和 profile 文件区别在于每次打开都会加载一次

其余配置脚本具体情况会因发行版有所不同，一般通过上述三种方式修改即可

1. https://www.cnblogs.com/keegentang/p/10671471.html
2. https://unix.stackexchange.com/questions/117467/how-to-permanently-set-environmental-variables
3. https://unix.stackexchange.com/questions/50665/what-is-the-difference-between-interactive-shells-login-shells-non-login-shell
4. https://www.cnblogs.com/lichihua/p/11574966.html

#### 管道
命令处理的结果可以通过管道 `|` 传到下个命令中

#### 逻辑
- `&`：同时运行左右侧两个命令，右侧没命令则表示后台运行
- `&&`：左侧命令返回正常时运行右侧命令
- `||`：左侧命令返回异常时运行右侧命令

#### 版本
`#!/bin/bash` 和 `#!/bin/sh`：都是用于注明脚本使用的解释器，一般来说，`sh` 是一种 POSIX，也是最开始的 shell，`bash` 则是初版 shell 的升级版。大概是未来确保兼容性，`sh` 作为 POSIX 依然在后续保留，可以通过软连接到 `bash` 来实现（现在一般使用 `bash` 的缩小移植版 `dash`），在语法等规则上和 `bash` 有所不同。总的来说，如果不考虑兼容问题，用 `bash` 就好了。

## Shell 命令
#### 实用工具
- `eval $(command)`：对命令进行简单化，将参数合成一个字符串，用结果作为 shell 的输入，并且执行得到的命令。就是若命令中引用了其他变量，则 eval 会先找出具体变量值替换掉命令，然后再执行命令。
- 对于非法字符的文件/文件夹名：  
	- 在整体名字前加上 ` -- ` 即可处理 `-` 为前缀的文件名
	- 在字符前加上 `\` 可以处理转义字符
	- 整体加上 `' '` 可以处理空格等字符（转义也可以实现）
	- 支持类似正则表达式 `ls /dev/sda[0-9]`、`ls /dev/sda*`
- `grep`：文本搜索（支持张泽）
	- 使用 EREs 正则 `grep -E "loss: \w+"`
	- 只输出匹配内容 `grep -o "xxx"`
- `awk`：文本分析工具（支持正则）
	- 使用方式：`awk 'pattern {action}' filename`（单双引号有所不同）
	- 以空白符为分隔符，打出第十四个字符串：`cat nas_model_100e_24bs.log | grep " - val_loss:" | awk  '{printf("%s,", $14)}'`，若打印为单个整数，则需要 `%d`
	- `awk -F ':'` 指定分隔符为 `:`
	- pattern 正则：`~`，为匹配符，用法如：`ls -l | awk '$3 ~/root/ {print $9}'`，检查第三字段匹配是否匹配 `root`，若是则打印其第九字段，其中 `/` 为模式匹配标识？
- `wc`：统计字数
	- 默认参数显示 行数、字数，以及字节数，如统计文件夹下的文件数（不包含隐藏文件）：`ls | wc` 第一和第二个
- `du`：查询文件占用空间 disk usage
	- `-s`：只列出当前文件大小，不列出子目录（如果是目录就统计目录下所有文件）
	- 查询当前目录所有文件和文件夹大小+排序：`du -sh * | sort -hr`
	- `-d1`：统计深度为 1
- `head -6`：管道控制输出前 6 个
- `find`：搜索工具，比如使用 `find ./ -name "xxx_*"` 搜索当前路径下前缀名为 xxx_ 的文件
    - `-size`：指明搜索的文件大小，`-10M` 代表小于 10MB，`+4M` 代表大于 4MB，可以写两个表明范围
	- `-type`：指明搜索的文件类型，`f` 代表文件，`d` 代表目录
	- `--max-depth=1`：指明搜索深度，需要放在路径后面，其他选项的前面
- `tar`：归档工具，可以使用 gzip 附带压缩，压缩效果还行
	- 压缩打包：`tar cvzf xxx.tar.gz xxx`，多次打包可以用 `tar uvf xxx.tar xxx`，之后在一起 `gzip xxx.tar` 压缩
	- 解压缩：`tar xvzf xxx.tar.gz`，多核解压可用 `pigz -p 8 -d xxx.tgz`
	- `--exclude=`：排除一些文件，可指定多次，支持通配符，__要放在打包的目录前__，也就是 `tar cvzf xxx.tar.gz --exclude="asd" xxx`
- `7z`：压缩工具，支持正则表达式
    - `7z a -xr'!*/.git' work-${day}.zip work`
- `crontab`：定时任务工具
    - `crontab -e` 进入编辑界面，写好定时任务就行。因为是后台执行，所以似乎不能开图形界面的任务
    - 写法见 https://crontab.guru/examples.html
- `sleep`：延时工具，`sleep 3m` 延时三分钟
- `vlc`：命令行可用的媒体播放器
    - `vlc --no-video --random --play-and-stop ./`：随机播放本目录下的音乐
- `netstat`：查询端口占用情况
    - `netstat -tunlp`，tcp udp listen pid
- `nslookup`：测试 dns 用
    - `nslookup baidu.com 8.8.8.8`，表示从 8.8.8.8 dns 服务器查询网站 IP，缺省则使用当前 DNS
- `chown`：修改文件拥有者
    - `chown -R x:xgroup dir`，递归修改目录下为 x 用户和 xgroup 组

#### 后台挂起
在命令后加上一个单独的 `&` 表示开启后台进程运行，终端可以做其他的。若在命令之前加上 `nohup` 则终端关闭不影响正在运行中的命令（NO SIGHUP），另外还会将输出默认重定向到 `nohup.out` 文件里，详见： https://www.jianshu.com/p/747e0d5021a2

#### 输出重定向
```sh
./build/tools/caffe time -model='lenet_train_test_decompose.prototxt' \
-gpu=0 2>&1
```
命令行代码后面 `2>&1` 代表输出重定向, 1为标准输入 ( 即终端 ), 2为标准输出, 3为标准错误。一般用于将 C/C++ 输出重定向至终端, 方便在后面加 `**** 2>&1 | tee output.log` 将输出另存  

如果打算抛弃掉标准输出或标准错误的话，可以把对应输出重定向到 `/dev/null`

```sh
echo -e "[xx]\nuiep" >> xxx.conf
```
使用 `echo` 指令可以将一些设置字符串加入到文件中，使用 `>>` 表示添加到文件末尾（`>` 则是重新建立个文件覆盖原有文件），`-e` 表示启用转义字符

#### 提权
- `sudo`：常用的普通用户获取权限命令，会切换环境变量，如需保持用户环境变量则需要加 `-E`

## Shell 变量
shell 变量为全局, 直接赋值后面就能用  

#### 输入
- 通过参数传入，如 `xxx = $1`，`xxxx = $2`。另外 `$#` 代表入参个数（不包括 `$0` 执行文件名），使用 `"$@"` 会返回输入参数的列表
- 通过 `$(command)` 传入，如 `xxx = $(command)`，这里 `$()` 意为对里面内容做替换并执行命令（ 类似 `eval` ），即将命令运行结果作为变量输入。\` \` 也能完成一样的事，但是需要处理转义字符，`$()` 不是所有类 Unix 系统都支持
- 通过 `$(( a+b ))` 传入，这里 `$(( ))` 用于整数运算，支持其他进制运算，十进值作为输出

#### 引用
- 使用 `${xxx}` 引用，支持替换和截断等一些操作
- 直接 `$xxx` 引用
- 双引号引用 `"${xxx}"` 表示变量内容看作字符串，否则为空时当作无输入（或者存在空格时 if 判断报错）

#### 整型变量
	A=15  
	A=$((A+1))  # 注意 = 号左右不能有空格

#### 数组
- 用括号定义，空格隔开 `a=(A B "C" D)`，下标索引 `${a[0]}`
- 获取长度用 `${#a[@]}`
- 将字符串转成数组需要注意有没空格，因为默认分割符号空格，也可以通过  `IFS="|"` 来修改分割符号为 `|`

#### 常用变量
```sh
# 脚本所在路径
SHELL_FOLDER=$(dirname $(readlink -f "$0"))
```

## Shell 脚本
#### 条件判断
格式：
```sh
if [ -d xxx ]; then
    xxx
elif [ -d xxxx ]
    xxx
else
    xxx
fi
```
flags：
```sh
if [ -e xxx ] # xxx 是否存在
if [ -d xxx ] # xxx 是否为目录
if [ -f xxx ] # xxx 是否是常规文件
if [ -L xxx ] # xxx 是否符号链接
if [ -r xxx ] # xxx 是否可读
if [ $x == $y ] # 判断相同与否
if [ $x = $y ]  # 判断相同与否，单等号也一样，在算数表达式中 (( )) 则不同（为赋值）
if [ "$a"x != ""x ]  # 将变量 a 看做字符串，然后判空，后面的 x 是为了让空字符串能拿来判断
if [[ $x == *$y ]] # 支持正则化表达 ( x末尾字符串是否为y )
if [[ "$root_dir" =~ ${expression} ]]  # 正则表达式匹配，${expression} 不能加双引号
if [ "$#" -lt 2 ] # 判断入参数是否小于 2 
```
#### 循环
格式：
```sh
for class in $(ls ${current_dir})
    do
    done
    
while [ ]:
    do
    done

for(( i=0;i<${#exp_name[@]};i++ ))
do
    echo ${exp_name[${i}]}
done
# 数组循环
```
还可以类 C 循环
```sh
for((i=1;i<=5;i++));do
    echo "这是第 $i 次调用";
done;
```

#### 对比同文件名
```sh
for file in `find images -type f | awk -F "/" '{print $NF}'`;do find masks -type f -iname "\$file" | awk -F "/" '{print $NF}';done
```

#### 快速删除
```sh
#!/bin/bash
for file in $(ls ./)
do 
    rm -r ${file} &
done
```

#### 字符串截断
```sh
# 转义字符记得加 \
${filename}           原字符串
${filaname#*/}        去除从左数('#') 第一个'/'及其左边的所有字符
${filename##*/}       去除从左数('#') 最后一个'/'及其左边的所有字符
${filename%/*}        去除从右数('%') 第一个'/'及其右边的所有字符
${filename%%/*}       去除从右数('%') 最后一个'/'及其右边的所有字符

${str/${a}/${b}}      将字符串 'str' 中的第一个 ${a} 转为 ${b}

# 仅用于文件？
cut -d. -f1           以 . 为分割符分割字符串，取左边的字符串
sed 's/要被取代的字串/新的字串/g'
```


#### 函数
```sh
#!/bin/bash
# /bin/sh 无法解析函数
function dir_recursion(){
  for file in $(ls $1)
    do
      if [ -d $1"/"$file ]; then
        dir_recursion $1"/"$file
      else
        suffix=${file##*.}
        if [ "${suffix}"x = "${suffix1}"x ]||[ "${suffix}"x = "${suffix2}"x ]; then
          echo $1/${file} >> images.txt
        fi
      fi
    done
}
dir_recursion $root_dir
```
递归调用可实现遍历

## 实用示例

#### 获取脚本的绝对路径

```sh
# 获取脚本所在路径，readlink 考虑符号链接
SHELL_FOLDER=$(dirname $(readlink -f "$0"))

# 添加环境变量
CUR_PATH=$(cd $(SHELL_FOLDER); pwd)
```