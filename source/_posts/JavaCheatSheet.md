---
title: Java Cheatsheet
date: 2020-04-09 11:50:24
tags: 
- 语法
categories:
- Java
---
# Background 
This post contains some frequently used knowledge in Java.

# Comparator
```java
public class NameSorter implements Comparator<Employee>
{
    @Override
    public int compare(Employee e1, Employee e2) {
        return e1.getName().compareToIgnoreCase( e2.getName() );
    }
}
```
The purpose of annotation @Override, referenced from Dave L.'s answer.
https://stackoverflow.com/questions/94361/when-do-you-use-javas-override-annotation-and-why 
<!--more-->
Use it every time you override a method for two benefits. Do it so that you can take advantage of the compiler checking to make sure you actually are overriding a method when you think you are. This way, if you make a common mistake of misspelling a method name or not correctly matching the parameters, you will be warned that you method does not actually override as you think it does. Secondly, it makes your code easier to understand because it is more obvious when methods are overwritten.

## Arrays.sort()
referenced from Java Document, ```sort(T[] a, Comparator<? super T> c)
Sorts the specified array of objects according to the order induced by the specified comparator.```
+ Example
```java
  Arrays.sort(p, (l1, l2) -> l2[0] == l1[0] ? l1[1] - l2[1] : l1[0] - l2[0]);
```

# Random
```java
Random rand = new Random()
targ = rand.nextInt(N) // generate a integer within [0, N)
```

# Data Structure
## Min heap
```java
PriorityQueue<Integer> pQueue = new PriorityQueue<Integer>(); 
pQueue.peek()
```

## HashMap
### Sort by Key
```java
TreeMap
```

### Sort by Value
```java
// (1) create a compartor
// (2) collect the entryset and turn it into an arraylist
// (3)sort the list based on the comparator
Comparator<Entry<Integer, Integer>> valueComparator = new Comparator<Entry<Integer,Integer>>() {
  @Override public int compare(Entry<Integer, Integer> e1, Entry<Integer, Integer> e2) { 
    Integer v1 = e1.getValue(); Integer v2 = e2.getValue(); return v1.compareTo(v2) * -1;
    } 
};
Set<Entry<Integer, Integer>> entriesA = mapA.entrySet();
List<Entry<Integer, Integer>> listOfEntriesA = new ArrayList<Entry<Integer, Integer>>(entriesA);
Collections.sort(listOfEntriesA, valueComparator);
```