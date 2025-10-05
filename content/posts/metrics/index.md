---
title: "The Importance of (Good) Metrics"
date: 2025-10-03
tags: ["machine learning", "gaussian processes", "probability", "stochastic processes", "kernel methods"]
author: "Daniel López Montero"
showToc: true
draft: true
description: "Math and Code"
ShowWordCount: false
ShowReadingTime: true
comments: true
UseHugoToc: true
editPost:
    URL: "https://github.com/dani2442/dani2442.github.io/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

At first I wanted this post to be only about metrics in machine learning, but this concept is more universal and does not make sense to study it as a separate problem. The ultimate goal of this post is to make you think and reflect.

We live surrounded with metrics: grades by teachers, employers, number of publications in Academia, FLOPs in computing, ELO in chess, salary, IQ for intelligence, movies rating in IMDB, books with GoodReads, things we buy in Amazon, election results in democracy, F1 score in machine learning, GDP for countries, EBITDA in finance, likes/followers in instagram, time in TikTok for content recommendation algorithms, etc etc.

In most cases these metrics try to summarize a copmlex variable system in a single number, which in most cases is inacurate. For example, a well-behaved metric must be comparable, i.e., 

> if A>B and B>C, then A>C

In mathematics, this property is called *transitivity*. However, in real life this almost never the case, there are usually trade-offs.

And in some cases choosing a wrong metrics can lead to catastrophic failure (as we will see with some examples). I want to argue in this post that choosing the right metrics is as important as solving them, in a similar way that asking the correct questions is as important as solving them. Also, I want to give some advice in case of doubt.


Goodheart's law


- p-hacking in statistics
- overfitting
- During the USSR, company leaders cheating to improve bonus
- CEOs pulling financial tricks to get better bonuses next quarter instead of improving the company long-term
- in class: cheating, memorizing instead of understanding
- football: "ganar, ganar y ganar"
- academia: maximizing publications over quality
- private healthcare: when the goal of a hospital is not cure the patient but rather to make money, it can lead to a perverse incentive where they overcharge, use prolong the treatment, offer more expensive solutions to simple problems [[1]](#references). This also happened with the education system. 
- democracy: political parties only care about the results of the next elections and not the long-term prosperity or planning of the city. unaffordability of houses short-term thinking, etc.

This is all the same form of: "The end justifies the means" mentality.

These are a few examples I could came up with. But I want you to think of other examples where this pattern arises

Importance of Benchmarking and recent success in LLMs is due to very well-curated and rich benchmarks: bla bla

How can we I apply this to our daily life?
For everyone: The first think that came to my mind is how each of one measure success. We may be baised by other people and think that success is only money, status, etc. Lately each of us has to life according to your inner-self and not to life up to the expectations of others.

![alt text](image.png)

If you are a parent: set the right incentives to your children
If you are a teacher: set the right incentives to you students
If you are a leader/employer: set the right incentives to you employees


Charlie Monger: tell me your incentive and I will tell you the result


This almost seems trivial to say but if a single metric is not enough, use more than one. For example, in machine learning it is very frequent to use the confussion matrix instead of single digit metrics. Modern LLM benchmarking uses many different datasets to compare the models.


## References
[1] "I Was An MIT Educated Neurosurgeon Now I'm Unemployed And Alone In The Mountains How Did I Get Here?" https://www.youtube.com/watch?v=25LUF8GmbFU&t