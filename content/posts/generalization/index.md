---
title: "Notes on Generalization Theory"
date: 2026-01-04
tags: ["machine learning", "probability"]
categories: ["probability", "machine learning"]
author: "Daniel López Montero"
showToc: true
draft: false
description: "Generalization bounds and theory for machine learning models."
ShowWordCount: false
ShowReadingTime: true
comments: true
TocOpen: true
UseHugoToc: true
editPost:
    URL: "https://github.com/dani2442/dani2442.github.io/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---


How do we know if a machine learning model will perform well on unseen data? What happens if you continue to add samples to the dataset? Is it better to have a more complex model or a simpler one?

These questions have been around for many years and are central to the field of statistical learning theory. Generalization theory provides mathematical guarantees and bounds on the generalization capability of families of functions. I have prepared a few notes on the basics of generalization theory. The only prerequisite is probability theory, and it is intended to be self-contained. It includes the most important results, such as Dudley's Theorem and McDiarmid's Theorem.

[Link to pdf](/files/generalization.pdf)

![alt text](image.png)