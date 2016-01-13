#!/usr/bin/env python
#coding:utf-8
from hmm import HMM_SEG

if __name__ == '__main__':

	hmm = HMM_SEG('msr_training.utf8')
	text1 = "人们常说生活是一部教科书"
	text2 = "每天高高兴兴的看人民日报。"
	text3 = "你是个傻子吗？"
	text4 = "桂林山水甲天下"
	text5 = "绿茶中我最喜欢信阳毛尖。"
	text6 = "全国销量领先的红罐凉茶改成广州恒大。"
	text7 = "系统库提供的内部函数。"
	print hmm.decode(text1)
	print hmm.decode(text2)
	print hmm.decode(text3)
	print hmm.decode(text4)
	print hmm.decode(text5)
	print hmm.decode(text6)
	print hmm.decode(text7)
