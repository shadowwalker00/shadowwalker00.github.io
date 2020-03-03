---
title: LeetCode338题解 Counting Bits
date: 2018-10-25 09:35:47
tags: [动态规划, 位运算]
categories: LeetCode
---

<blockquote class="blockquote-center">一個人，或许真的孤單。或許，一個人的孤單，只是一種生活</blockquote>

这道题用到了为了复习一下位运算的操作符，特意用了一下位运算符。题目的提示其实给了一种简单版本的答案，就是统计每一个数字的one的个数。因此时间复杂度是$O(n\times sizeof(int))。进阶版本使用了DP动态规划，实际上就是找到数字之间的规律。

<!--more-->

# 简单版本

```c++
class Solution {
public:
    vector<int> countBits(int num) {
        vector<int> res(num+1);
        for(int i=0;i<=num;i++){
            res[i] = countOne(i);
        }
        return res;
    }
    int countOne(int num){
        int counter = 0;
        while(num>0){
            int tail = num%2;
            if(tail==1) counter++;
            num /= 2;
        }
        return counter;
    }
};
```

# 进阶版本

**思路：**观察一下，其实，一个整数的二进制前半部分（除最后一位），与这个整数一半对应的整数对应的二进制是一样的。举例来讲，整数11的二进制是1011，而其一半对应的整数是5，其二进制是101，能够看到11的前半部分和5的二进制都是101。原因是因为11的前半部分是通过将5进行逻辑左移的得到的，固然相等。所以，11前半部分one的个数就等于5中one的个数。而末尾则是直接mod2来判断是0还是1，进而决定加1或不加。

```c++
class Solution {
public:
    vector<int> countBits(int num) {
        vector<int> res(num+1,0);
        for(int i =1;i<=num;i++){                        
            res[i] = res[i/2] + (i&1);
        }
        return res;
    }
};
```

# 复习：位运算符

1. 逻辑与 &

运算通常用于二进制取位操作，例如一个数 &1的结果就是取二进制的最末位。

2. 逻辑或 |

| 运算通常用于二进制特定位上的无条件赋值，例如一个数|1的结果就是把二进制最末位强行变为1。

如果需要把二进制最末位变成0，对这个数 |1之后再减一就可以了，其实际意义就是把这个数强行变成最近接的偶数

3. 逻辑异或 ^

^运算通常用于对二进制的特定一位进行取反操作，^运算的逆运算是它本身