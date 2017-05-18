# finalproject

> Note: much of the language model and subreddit classification code is in the colin/baseline (bad branch name..) branch.

Computational Linguistics II Final Exam Project: Exploring Linguistic Signal for Suicidality in Social Media

```
hist(fperson$PosScore, breaks = 20, xlim=c(0,2), col=rgb(0,0,1,0.5))

hist(fperson$NegScore, breaks = 20, xlim=c(0,2), col=rgb(0,0,1,0.5), add=T)
```
# Readability and Time Frame implementation
Readability and Time frame code is included in Readability_timeframe.py code.
Code is written sequentially so just implementing it gives the sequential results
  -Runs the code for checking the Readability score distribution for all posts and for user level distribution
  -Gives t-test values(statistics and p-value) for readability score of positve and control users
  -Runs code for checking time frame distribution for all posts and for user level distribution
  -Gives t-test values for time frames of positive and control users
  -trains the classifiers with train data and tests the classifier against test data.
  
 Final output is Precision, recall and f-1 scores along with confusion matrix for time-frame as feature. 


### Colin's R commands

```
# d = "word-n-gram/"
d = "char-n-gram/"
pp = read.table(paste(d, "pos.pos.txt",sep=""))
pn = read.table(paste(d, "pos.neg.txt",sep=""))
pd = read.table(paste(d, "pos.delta.txt",sep=""))

np = read.table(paste(d, "neg.pos.txt",sep=""))
nn = read.table(paste(d, "neg.neg.txt",sep=""))
nd = read.table(paste(d, "neg.delta.txt",sep=""))

# word-level config
hist(as.numeric(pp$V1), breaks=c(seq(-50000,100,100)), col=rgb(0.5,0.5,1.0,0.5), xlab="positive language model log-prob", xlim=c(-3000,0), ylim=c(0,100))
hist(as.numeric(np$V1), breaks=c(seq(-50000,100,100)), col=rgb(1,.5,.5,0.5), xlab="positive language model log-prob", xlim=c(-3000,0), ylim=c(0,100),add=T)
legend("topleft", c("Positives", "Control"), col=c(rgb(0.5,0.5,1.0,0.5),rgb(1,.5,.5,0.5)), lwd=10)

hist(as.numeric(pn$V1), breaks=c(seq(-50000,100,100)), col=rgb(0.5,0.5,1.0,0.5), xlab="negative language model log-prob", xlim=c(-3000,0), ylim=c(0,140))
hist(as.numeric(nn$V1), breaks=c(seq(-50000,100,100)), col=rgb(1,.5,.5,0.5), xlab="negative language model log-prob", xlim=c(-3000,0), ylim=c(0,140),add=T)
legend("topleft", c("Positives", "Control"), col=c(rgb(0.5,0.5,1.0,0.5),rgb(1,.5,.5,0.5)), lwd=10)

hist(as.numeric(pd$V1), breaks=c(seq(-50000,100,100)), col=rgb(0.5,0.5,1.0,0.5), xlab="delta (pos-neg) language model log-prob", xlim=c(-3000,0), ylim=c(0,140))
hist(as.numeric(nd$V1), breaks=c(seq(-50000,100,100)), col=rgb(1,.5,.5,0.5), xlab="delta (pos-neg) language model log-prob", xlim=c(-3000,0), ylim=c(0,140),add=T)
legend("topleft", c("Positives", "Control"), col=c(rgb(0.5,0.5,1.0,0.5),rgb(1,.5,.5,0.5)), lwd=10)

# char-level config
hist(as.numeric(pp$V1), breaks=c(seq(-50000,100,5)), col=rgb(0.5,0.5,1.0,0.5), xlab="positive language model log-prob", xlim=c(-150,0), ylim=c(0,220))
hist(as.numeric(np$V1), breaks=c(seq(-50000,100,5)), col=rgb(1,.5,.5,0.5), xlab="positive language model log-prob", xlim=c(-150,0), ylim=c(0,220),add=T)
legend("topleft", c("Positives", "Control"), col=c(rgb(0.5,0.5,1.0,0.5),rgb(1,.5,.5,0.5)), lwd=10)

hist(as.numeric(pn$V1), breaks=c(seq(-50000,100,5)), col=rgb(0.5,0.5,1.0,0.5), xlab="negative language model log-prob", xlim=c(-150,0), ylim=c(0,220))
hist(as.numeric(nn$V1), breaks=c(seq(-50000,100,5)), col=rgb(1,.5,.5,0.5), xlab="negative language model log-prob", xlim=c(-150,0), ylim=c(0,220),add=T)
legend("topleft", c("Positives", "Control"), col=c(rgb(0.5,0.5,1.0,0.5),rgb(1,.5,.5,0.5)), lwd=10)

hist(as.numeric(pd$V1), breaks=c(seq(-50000,100)), col=rgb(0.5,0.5,1.0,0.5), xlab="delta (pos-neg) language model log-prob", xlim=c(-30,0), ylim=c(0,220))
hist(as.numeric(nd$V1), breaks=c(seq(-50000,100)), col=rgb(1,.5,.5,0.5), xlab="delta (pos-neg) language model log-prob", xlim=c(-30,0), ylim=c(0,220),add=T)
legend("topleft", c("Positives", "Control"), col=c(rgb(0.5,0.5,1.0,0.5),rgb(1,.5,.5,0.5)), lwd=10)

wilcox.test(pp$V1, np$V1)
wilcox.test(pn$V1, nn$V1)
wilcox.test(pd$V1, nd$V1)
```
