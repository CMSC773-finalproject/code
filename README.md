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

hist(as.numeric(pp$V1), breaks=c(seq(-50000,100,100)), col=rgb(0.5,0.5,1.0,0.5), xlab="positive language model log-prob", xlim=c(-3000,0), ylim=c(0,100))
hist(as.numeric(np$V1), breaks=c(seq(-50000,100,100)), col=rgb(1,.5,.5,0.5), xlab="positive language model log-prob", xlim=c(-3000,0), ylim=c(0,100),add=T)
legend("topright", c("Positives", "Control"), col=c(rgb(0.5,0.5,1.0,0.5),rgb(1,.5,.5,0.5)), lwd=10)

hist(as.numeric(pn$V1), breaks=c(seq(-50000,100,100)), col=rgb(0.5,0.5,1.0,0.5), xlab="negative language model log-prob", xlim=c(-3000,0), ylim=c(0,140))
hist(as.numeric(nn$V1), breaks=c(seq(-50000,100,100)), col=rgb(1,.5,.5,0.5), xlab="positive language model log-prob", xlim=c(-3000,0), ylim=c(0,140),add=T)
legend("topright", c("Positives", "Control"), col=c(rgb(0.5,0.5,1.0,0.5),rgb(1,.5,.5,0.5)), lwd=10)

hist(as.numeric(pd$V1), breaks=c(seq(-50000,100,100)), col=rgb(0.5,0.5,1.0,0.5), xlab="negative language model log-prob", xlim=c(-3000,0), ylim=c(0,140))
hist(as.numeric(nd$V1), breaks=c(seq(-50000,100,100)), col=rgb(1,.5,.5,0.5), xlab="delta (pos-neg) language model log-prob", xlim=c(-3000,0), ylim=c(0,140),add=T)
legend("topright", c("Positives", "Control"), col=c(rgb(0.5,0.5,1.0,0.5),rgb(1,.5,.5,0.5)), lwd=10)

wilcox.test(pp$V1, np$V1)
wilcox.test(pn$V1, nn$V1)
wilcox.test(pd$V1, nd$V1)
```
