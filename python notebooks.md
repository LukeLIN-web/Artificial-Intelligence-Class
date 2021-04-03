# python笔记



动态类型, 不用声明, 变量类型可以随时改变.

是一个oop的语言.

### 数据类型

basic type : int/float/ complex , str . bool

container :  list/tuple,  dict/ set 

dict是用来存储键值对结构的数据的，set其实也是存储的键值对，只是默认键和值是相同的。Python中的dict和set都是通过散列表来实现的

tuple ,int , str是immutable的,值传递  list,dict是 mutable的, 引用传递.

list 的方法有: insert, remove, slice, index, interate,  

range(10)  产生一个0到9的 range object.

`list(range(10));`

终止index是 左闭右开的. 

TypeError: 'int' object is not subscriptable此报错一般是在整数上加了下标：

#### list

list中加入tuple

```python
res += (i, j)是不对的, 就是加了两个int
res.append((i, j)) correct
xy = (i, j)
res.append(xy) correct
```

list中find:  if a in list :就可以了.

遍历list

list 中元素个数 : open_list.__len__()

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
2.np.where(condition) 当where内只有一个参数时，那个参数表示条件，当条件成立时，where返回的是每个符合condition条件元素的坐标,返回的是以元组的形式.

eq怎么用?

```python
print``(cat_1.__eq__(cat_2)) 
```

==  运算符是比较哪些?

- `is`比较的是两个整数对象的id值是否相等，也就是比较两个引用是否代表了内存中同一个地址。
- `==`比较的是两个整数对象的内容是否相等，使用`==`时其实是调用了对象的`__eq__()`方法。

对于整数对象，Python把一些频繁使用的整数对象缓存起来，保存到一个叫small_ints的链表中，在Python的整个生命周期内，任何需要引用这些整数对象的地方，都不再重新创建新的对象，而是直接引用缓存中的对象。Python把频繁使用的整数对象的值定在[-5, 256]这个区间，如果需要这个范围的整数，就直接从small_ints中获取引用而不是临时创建新的对象。因为大于256或小于-5的整数不在该范围之内，所以就算两个整数的值是一样，但它们是不同的对象。

所以python会显示:  `257 is not 257`

通过比较它的b属性建立堆排序::



```python
1.   def __lt__(self, other):
2.         if self.b<other.b:
3.             return True
a = P(3,1)
b = P(2,3)
c = P(10,0)
d = P(3,1)
 
h = []
heapq.heappush(h,a)
heapq.heappush(h,b)
heapq.heappush(h,c)
heapq.heappop(h)
就会把10,0 排在第一个,弹出
```



### 切片

https://www.liaoxuefeng.com/wiki/1016959663602400/1017269965565856

 取前3个元素`L[0:3]` 左闭右开

```python
>>> L[-2:] 后两个数
['Bob', 'Jack']
>>> L[-2:-1]
['Bob']
前10个数，每两个取一个：
>>> L[:10:2]
[0, 2, 4, 6, 8]
所有数，每5个取一个：
>>> L[::5]
childs.append(curr_state[:idx - 1] + curr_state[idx:idx + 1] + curr_state[idx - 1:idx] + curr_state[idx + 1:]) 
两行交换, 越界了也不会报错.
交换二维list中两个元素,和np中array不一样.
      curr_state.state[:row] + [
            curr_state.state[row][:col] + curr_state.state[row + 1][col:col + 1] + curr_state.state[row][col + 1:]] + [curr_state.state[row + 1][:col] + curr_state.state[row][col:col + 1] + curr_state.state[row + 1][ col + 1:]] + curr_state.state[row + 2:]
  
```





### 垃圾回收

python采用的是引用计数机制为主，标记-清除和分代收集（隔代回收）两种机制为辅的策略。

PyObject是每个对象必有的内容，其中ob_refcnt就是做为引用计数。当一个对象有新的引用时，它的ob_refcnt就会增加，当引用它的对象被删除，它的ob_refcnt就会减少

#### **导致引用计数+1的情况**

- 对象被创建，例如a=23
- 对象被引用，例如b=a
- 对象被作为参数，传入到一个函数中，例如`func(a)`
- 对象作为一个元素，存储在容器中，例如`list1=[a,a]`

#### **导致引用计数-1的情况**

- 对象的别名被显式销毁，例如`del a`
- 对象的别名被赋予新的对象，例如`a=24`
- 一个对象离开它的作用域，例如:func函数执行完毕时，func函数中的局部变量（全局变量不会）
- 对象所在的容器被销毁，或从容器中删除对象

#### **分代回收**

- 分代回收是一种以空间换时间的操作方式，Python将内存根据对象的存活时间划分为不同的集合，每个集合称为一个代，Python将内存分为了3“代”，分别为年轻代（第0代）、中年代（第1代）、老年代（第2代），他们对应的是3个链表，它们的垃圾收集频率随着对象存活时间的增大而减小。
- 新创建的对象都会分配在**年轻代**，年轻代链表的总数达到上限时，Python垃圾收集机制就会被触发，把那些可以被回收的对象回收掉，而那些不会回收的对象就会被移到**中年代**去，依此类推，**老年代**中的对象是存活时间最久的对象，甚至是存活于整个系统的生命周期内。
- 同时，分代回收是建立在标记清除技术基础之上。分代回收同样作为Python的辅助垃圾收集技术处理那些容器对象



交换

Python中没有swap()函数,交换两个数的方式

```python
a,b = b,a
```



pycharm 一键注释: ctrl +/



pycharm无法最大化, pycharm最小化打不开。

解决方法: 重装pycharm也不行, 换个项目就可以了.



#### numpy库

```python
np.zeros(shape=(4, 4)) 
```



 ValueError: operands could not be broadcast together with shapes (0,) (3,)



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