语料下载地址：

[**WIT语料库**](https://wit3.fbk.eu/mt.php?release=2015-01)

选择其中中英部分的语料下载解压就行，我们暂时只需要其中的两个文件：`trian.tags.en-zh.en`和`trian.tags.en-zh.zh`，
两个文件中的内容是逐行一一对应的。



## 切词和清洗
有了语料之后的第一件事情就是对语料进行清洗和基本的切词操作了。
对于英文的切词，这里使用[**moses**](https://github.com/moses-smt/mosesdecoder)工具中的`tokenizer.perl`，我们这里把这个文件拿过来放在
本项目的`utils`文件夹下面。

