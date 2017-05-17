# finalproject

Computational Linguistics II Final Exam Project: Exploring Linguistic Signal for Suicidality in Social Media

```
hist(fperson$PosScore, breaks = 20, xlim=c(0,2), col=rgb(0,0,1,0.5))

hist(fperson$NegScore, breaks = 20, xlim=c(0,2), col=rgb(0,0,1,0.5), add=T)
```

### Colin's R commands

```
pp = read.table("pos.pos.txt")
pn = read.table("pos.neg.txt")
pd = read.table("pos.delta.txt")

np = read.table("neg.pos.txt")
nn = read.table("neg.neg.txt")
nd = read.table("neg.delta.txt")

hist(as.numeric(pp$V1), breaks=c(seq(-8000,100,100)), col=rgb(0.5,0.5,1.0,0.5), xlab="positive language model log-prob", xlim=c(-3000,0), ylim=c(0,20))
hist(as.numeric(np$V1), breaks=c(seq(-8000,100,100)), col=rgb(1,.5,.5,0.5), xlab="positive language model log-prob", xlim=c(-3000,0), ylim=c(0,20),add=T)
```
