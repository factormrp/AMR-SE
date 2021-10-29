# Analysing Mathematical Reasoning -> Student Edition
This mini-project is an attempt at grasping the concept of a transformer. In particular, I used the [mathematics_dataset](https://github.com/deepmind/mathematics_dataset) released by Google DeepMind as part of their ICLR 2019 paper Analysing Mathematical Reasoning Abilities of Neural Models. I attempt to validate the results found therein by leveraging the [transformer model](https://pytorch.org/tutorials/beginner/transformer_tutorial.html) proposed by Pytorch.

## Inspiration
The main reason I took on this was because I had been discussing the idea of creating a model that could process and understand mathematical text prompts and stumbled upon the DeepMind work. Further digging showed us that I was about a year late in my thinking but the idea was still fresh. Then we found a [Stanford students'](http://cs230.stanford.edu/projects_fall_2019/reports/26258425.pdf) final report paper on their validation attempt. Feeling confident this would be a great way to step into the pool, this is my attempt.

The model processes the sequence and converges to a minimized point. However, I could only implement cross-entropy loss and could not find a way at the time to convert the mertric to accuracy. This reflects a working knowledge of implementing models but failing to fully understand the underlying procedures at play.
