---
title: "The Importance of (Good) Metrics"
date: 2025-10-03
tags: ["machine learning", "metrics"]
categories: ["miscellaneous", "machine learning"]
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

And in some cases choosing the wrong metrics can lead to catastrophic failure (as we will see with some examples). I want to argue in this post that choosing the right metrics is as important as solving them, in a similar way that asking the correct questions is as important as solving them. Also, I want to give some advice in case of doubt.


## Egregrious examples of bad metrics
- p-hacking in statistics: it is a commonplace in statistics to use p-values to describe how confident we are at the results of some statistics test. The standard threshold use to say when a result is *statistically significant* is 0.05, this has led many *scientist* to aim for this arbitrary threshold at all cost. We have seen numerous retractions over the years and i assume uncountable are never noticed [[2]](#references).

- overfitting in machine learning: over the years, there has been many examples of *data leaking*, i.e., training the model using the testing set split [[3]](#references). Also, when your only goal is to optimize for a single metric, you may overfit to that single task, even without using the test, leading to a model that is virtually useless. 

- Cooking the books in the USSR: In the 20th century, During the USSR under Khrushchev, company and factory leaders often manipulated production data or falsified results to appear more successful than they actually were. Because the Soviet economic system rewarded meeting targets rather than genuine efficiency or quality, managers frequently inflated numbers, hid waste, or produced useless goods just to report success. This and other causes led to the fall of the USSR.

- Incentives for directives: CEOs were usually payed bonuses based on the results during the earnings each quarter. So many CEOs prioritizes short-term wins over long-term goals for healthy company. They pulled financial tricks and this led to many poor performance for these companies. Consequently, one of the most well-known quotes in investing nowadays is: "Show me the incentive and I'll show you the outcome" attributed to Charlie Munger. 

- in class: although the main goal of going to school is learning, to way to measure is usually quite different, through exams or projects. And probably you or many people you know practice the *art of cheating*. Another way of cheating is memorizing instead of understanding, which led to poor learning and to forget concepts after the exam.
- football: "ganar, ganar y ganar"

- academia: the most important metric for researchers is the number of publications and citations. This created a big misalignment of goals and nowadays, there is a broken culture were publishing many and fast is the norm [[4]]. The following graphs show the evolution of the number of publications and average cites per paper [[6, 7]](#references). It seems that we are producing so much information no one reads...

![alt text](image-1.png)

- private healthcare: when the goal of a hospital is not cure the patient but rather to make money, it can lead to a perverse incentive where they overcharge, use prolong the treatment, offer more expensive solutions to simple problems [[1]](#references). This also happened with the education system. 

- democracy: political parties only care about the results of the next elections and not the long-term prosperity or planning of the city. unaffordability of houses short-term thinking, etc.


This is all the same form of: "The end justifies the means" mentality.
This has all be a byproduct of a disalignmnet of incentives. It is often attributed to this law:

> **Goodheart's law**\
> "when a measure becomes a target, it ceases to be a good measure"

However, i don't agree at all. I think that if your metrics and incentives are good enough, an equilibrium will follow. 



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

[2] p-hacking: https://en.wikipedia.org/wiki/Data_dredging

[3] data leaking: https://en.wikipedia.org/wiki/Leakage_(machine_learning)

[4] "Scientific publishing is broken" https://www.theguardian.com/science/2025/jul/13/quality-of-scientific-papers-questioned-as-academics-overwhelmed-by-the-millions-published

[5] "The misalignment of incentives in academic publishing and implications for journal reform" https://www.pnas.org/doi/10.1073/pnas.2401231121

[6] Monteiro, Maria & Séneca, Joana & Magalhães, Catarina. (2014). The History of Aerobic Ammonia Oxidizers: from the First Discoveries to Today. Journal of microbiology (Seoul, Korea). 52. 537-47. 10.1007/s12275-014-4114-0. 

[7] Mike Thelwall, Pardeep Sud; Scopus 1900–2020: Growth in articles, abstracts, countries, fields, and journals. Quantitative Science Studies 2022; 3 (1): 37–50. doi: https://doi.org/10.1162/qss_a_00177