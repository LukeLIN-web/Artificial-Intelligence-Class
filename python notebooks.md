# python笔记



动态类型, 不用声明, 变量类型可以随时改变.

是一个oop的语言.

### 数据类型

basic type : int/float/ complex , str . bool

container :  list/tuple,  dict/ set 

tuple ,int , str是immutable的,值传递  list,dict是 mutable的, 引用传递.

list 的方法有: insert, remove, slice, index, interate,  

range(10)  产生一个0到10的 range object.

`list(range(10));`

终止index是 左闭右开的. 

TypeError: 'int' object is not subscriptable此报错一般是在整数上加了下标：

#### list

list中加入tuple

```python
res += (i, j)
```



##### 错误

1 : Shadows built-in name 'list' 

意思是你的你的变量名起的不好；最好要具体有意思，不要太随意，起像str、list、len等太随意的名字。



#### dict

大括号来产生一个dict

例如`band = {'drum': 'saya' , 'voacal' : 'kani'}`

所有mutable object是不能做key的.



#### tuple

不可变的序列, 





### 运算符

没有++ -- 

没有? :

 可以用两个乘表示平方, 两个除 表示整除, 可以用and ,or 

in 判断名字是否存在list之中.

is  判断是否为同一个对象.

is和 == 不同, 

pass // 还没想好写啥, 先pass.

for-in  for语句不需要int i 变量,

`for it in s`   类似于 cpp中的 `for(auto it : s)`

 可以同时返回多个值.





### 类

```python
class Circle:
    def __init__(self,v):
        self.__value=v
    def getArea(self):
        return 3.1415926*self.__value**2
    def getPerimeter(self):
        return 3.1415926*self.__value*2
c=Circle(5)
```

1.np.where(condition,x,y) 当where内有三个参数时，第一个参数表示条件，当条件成立时where方法返回x，当条件不成立时where返回y
2.np.where(condition) 当where内只有一个参数时，那个参数表示条件，当条件成立时，where返回的是每个符合condition条件元素的坐标,返回的是以元组的形式





## 报错

【python报错】Non-ASCII character '\xe5' 

解决方法：
在Python源文件的最开始一行，加入一句：

coding=UTF-8 或者 -*- coding:UTF-8 -*-

##### [python报错]"IndentationError: unexpected indent"的两三解决方法

这个是缩进错误，我们可以通过下面几步解决他：
首先检查代码是不是有错误的索引
如果没有，全都正确，可以看看是不是使用'''进行了整段的注释，如果是，一定要保证其与上下相邻代码缩进一致，而#就无所谓
如果还有错，使用notepad++打开文件，选择视图->显示符号->显示空格和制表符，然后查看是不是有空格与制表符混用的情况
vim可以用: set list 显示空格和制表符.
unexpected indent 就是说“n”是一个“意外的”缩进。也就是说，这里的问题就是指“n”是一个意外的缩进。通过查看源代码可知这里的确是缩进了一个字符位。
据此推断，我们把这句话的缩进取消，也就是顶格写，

##### [python报错]出现了AttributeError: object 'L2Cache' has no attribute 'connectCPUSideBus'

  (C++ object is not yet constructed, so wrapped C++ methods are unavail
对象“l2cache”没有属性 ,很多是说不要用跟系统库同样名字,这里则是因为之前的顶格写,导致没有定义到class中去.

##### 【python报错】TypeError: super(type, obj): obj must be an instance or subtype of type

class FooChild(FooParent):
    def __init__(self):
         super(FooChild,self)
 #首先找到 FooChild 的父类（就是类 FooParent），然后把类 FooChild 的对象转换为类 FooParent 的对象

##### Sccons

Scons是一个开放源码、以Python语言编码的自动化构建工具，可用来替代make编写复杂的makefile。并且scons是跨平台的，只要scons脚本写的好，可以在Linux和Windows下随意编译。
在Java的集成开发环境中，比如Eclipse、IDEA中，有常常有三种与编译相关的选项Compile、Make、Build三个选项。这三个选项最基本的功能都是完成编译过程。但又有很大的区别，区别如下：
1、Compile：只编译选定的目标，不管之前是否已经编译过。

2、Make：编译选定的目标，但是Make只编译上次编译变化过的文件，减少重复劳动，节省时间。（具体怎么检查未变化，这个就不用考虑了，IDE自己内部会搞定这些的）

3、Build：是对整个工程进行彻底的重新编译，而不管是否已经编译过。Build过程往往会生成发布包，这个具体要看对IDE的配置了，Build在实际中应用很少，因为开发时候基本上不用，发布生产时候一般都用ANT等工具来发布。Build因为要全部编译，还要执行打包等额外工作，因此时间较长。