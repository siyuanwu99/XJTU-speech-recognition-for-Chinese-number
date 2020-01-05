# ASR Course Work

This is a python implementation for speech recognition for Chinese numbers.
It can recognize number from 0 to 9 in chinese.
这段代码实现了音频数字0~9的分类识别。

[中文版](README-zh-CN.md)

## Environment

- python == 3.7
- numpy
- matplotlib
- sklearn
- librosa

## Introduction

- First extract time-series characters from frames of time domain & frequency domain features. 
- Then extract MFCC features using decimation-in-time radix-2 FFT algorithms
- Then use some basic machine learning algorithms (such as SVC, decision trees, etc.) to train a classification model
- Output accuracy and Confusion Matrix on test set
- Using ECOC methods to improve generalization ability

