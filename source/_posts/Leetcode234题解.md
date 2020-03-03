---
title: Leetcode234题解
date: 2018-10-23 22:09:54
tags: LeetCode
categories: LeetCode
---
<blockquote class="blockquote-center">优秀的人，不是不合群，而是他们合群的人里面没有你</blockquote>
Leetcode234是一个Easy题目，但是由于进阶要求空间复杂度是O(1)，因此我们撰写了这篇文章。主要用到了两个比较常用的链表算法:
1. 快慢指针。用于判断链表中是否存在环，以及求链表的中点
2. 反转链表。
<!-- more --> 

### 一般解法
时间复杂度O(n)，空间复杂度是O(n)

```c++
class Solution {
public:
    bool isPalindrome(ListNode* head) {
        ListNode *q=head;
        vector<int> temp;
        while(q!=NULL){
            temp.push_back(q->val);
            q = q -> next;
        }
        if(temp.size()==0) return true;
        int h = 0, r = temp.size()-1;
        while(h<r){
            if(temp[h]!=temp[r])  return false;
            else{
                h++;
                r--;
            }
        }
        return true;
    }
};
```

### 进阶解法
时间复杂度O(n)，空间复杂度O(1)

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    bool isPalindrome(ListNode* head) {
        if(!head) return true; // empty list
        ListNode *mid = findMid(head); 
        mid = reverseList(mid);    
        ListNode *p1 = head,*p2 = mid;
        while(p1!=NULL&&p2!=NULL){
            if(p1->val!=p2->val) return false;
            else{
                p1 = p1->next;
                p2 = p2->next;
            }
        }
        return true;
        
    }
    ListNode* findMid(ListNode* head){
        ListNode *slow = head,*fast = head;
        while(fast!=NULL&&fast->next!=NULL){
            fast = fast->next->next;
            slow = slow->next;            
        }
        return slow;
    }
    ListNode* reverseList(ListNode* head){
        //reverse the list
        ListNode* prev = NULL;
        while(head!=NULL){
            ListNode* temp = head->next;
            head->next = prev;
            prev = head;
            head = temp;
        }
        return prev;
    }
};
```

