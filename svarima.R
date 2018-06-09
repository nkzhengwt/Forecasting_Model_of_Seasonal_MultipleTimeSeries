library("MTS")
setwd("E:/Intern/CICC")
x<-read.table("income4.csv",header=TRUE,sep=',')
x<-subset(x,select=-c(date,date2))
mystats<-function(x,na.omit=FALSE){
  if(na.omit)
    x<-x[!is.na(x)]
  m<-mean(x)
  n<-length(x)
  s<-sd(x)
  skew<-sum((x-m)^3/s^3)/n
  kurt<-sum((x-m)^4/s^4)/n-3
  return(c(n=n,mean=m,stdev=s,skw=skew,kurtosis=kurt))
}
sapply(x,mystats)
summary(x)
x[is.na(x)]<-0
Yili<-x[c(1:60),c(1,10)]
Yili<-diffM(Yili)
plct<-VARMA(Yili,p=1,q=1)
VARMApred(plct,h=5)
plct<-VAR(Yili,p=2)
plct<-sVAR(Yili,p=2)
plct<-sVARMA(Yili,order=c(1,1,2),sorder = c(1,0,1),s=4)

