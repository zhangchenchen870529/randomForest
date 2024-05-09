args <- commandArgs(T)

cat("make sure:\n    y and x must have the same sample id\n    train_y must have only two levels and will change to 0 and 1\n    test_y which not in train_y will change to 2\n    set marker_num 0 to compute automaticity\nnote:if x and y have different levels could lead to errors\n")

if (length(args) != 10) {
  stop("Rscript *.R [train_x] [train_y] [test_x] [test_y] [cv_fold] [cv_step] [cv_time] [marker_num] [prefix] [seed]\n")
}

train.x <- args[1]
train.y <- args[2]
test.x <- args[3]
test.y <- args[4]
cv.fold <- as.numeric(args[5])
cv.step <- as.numeric(args[6])
cv.time <- as.numeric(args[7])
marker.num <- as.numeric(args[8])
prefix <- args[9]
seed <- as.numeric(args[10])
# package
library(randomForest)

args <- commandArgs(F)
SD <- dirname(sub("--file=", "", args[grep("--file=", args)]))
# function
source(paste0(SD, "/rfcv1.R"))
source(paste0(SD, "/ROC.R"))

# data
train.x <- t(read.table(train.x,sep="\t",head=T,row.names=1,check.names=F,quote=""))
train.y <- read.table(train.y,sep="\t",head=T,row.names=1,check.names=F,quote="")
train.x<-train.x[pmatch(rownames(train.y),rownames(train.x)),]

train.y<-as.factor(train.y[,1])
train.l <- levels(train.y)
levels(train.y) <- 0:1

test.x <- t(read.table(test.x,sep="\t",head=T,row.names=1,check.names=F,quote=""))
test.y <- read.table(test.y,sep="\t",head=T,row.names=1,check.names=F,quote="")
test.x<-test.x[pmatch(rownames(test.y),rownames(test.x)),]
test.y<-as.factor(test.y[,1])
test.l <- levels(test.y)
levels(test.y) <- 0:1


# crossvalidation
pdf.dir <- paste0(prefix, "_randomForest.pdf")
pdf(pdf.dir, width = 21, height = 7)
par(mfrow = c(1, 3))


set.seed(seed)
train.cv <- replicate(cv.time, rfcv1(train.x, train.y, cv.fold = cv.fold, step = cv.step), simplify = F)

error.cv <- sapply(train.cv, "[[", "error.cv")
error.cv.rm <- rowMeans(error.cv)

id <- error.cv.rm < min(error.cv.rm) + sd(error.cv.rm)

if (marker.num == 0) {
  marker.num <- min(as.numeric(names(error.cv.rm)[id]))
}
matplot(train.cv[[1]]$n.var, error.cv, type = "l", log = "x", col = rep(1, cv.time), main = paste("select", marker.num, "Vars"), xlab = "Number of vars", 
  ylab = "CV Error", lty = 1)
lines(train.cv[[1]]$n.var, error.cv.rm, lwd = 2)
abline(v = marker.num, col = "pink", lwd = 2)

# pick marker by corossvalidation
marker.t <- table(unlist(lapply(train.cv, function(x) {
  lapply(x$res, "[", 1:marker.num)
})))
marker.t <- sort(marker.t, d = T)
names(marker.t) <- colnames(train.x)[as.numeric(names(marker.t))]
marker.dir <- paste0(prefix, "_marker.txt")
write.table(marker.t, marker.dir, col.names = F, sep = "\t", quote = F)
marker.p <- names(marker.t)[1:marker.num]

# train model
set.seed(3)
train.rf <- randomForest(train.x[, marker.p,drop=F], train.y, importance = T)

###########################################predict result
train.p <- predict(train.rf, type = "prob")
pr.dir <- paste0(prefix, "_randomForest_train_probability.txt")

train.pp=array(,c(nrow(train.p),3))

for(i in 1:nrow(train.p)){
        train.pp[i,1]=train.p[i,2]
        if(train.pp[i,1]>=0.5){
                train.pp[i,2]=1
        }
        else{
                train.pp[i,2]=0
        }
        train.pp[i,3]=as.numeric(as.character(train.y[i]))
}
rownames(train.pp)=rownames(train.p)
colnames(train.pp)=c("predict_probability","predict_group","true_group")

write.table(train.pp, pr.dir, sep = "\t", quote = F, col.names = NA)


# train ROC
plot_roc(train.y, train.p[, 2])

# test predict
test.p <- predict(train.rf, test.x, type = "prob")

pr.dir <- paste0(prefix, "_randomForest_test_probability.txt")

test.pp=array(,c(nrow(test.p),3))
for(i in 1:nrow(test.p)){
        test.pp[i,1]=test.p[i,2]
        if(test.pp[i,1]>=0.5){
                test.pp[i,2]=1
        }
        else{
                test.pp[i,2]=0
        }
        test.pp[i,3]=as.numeric(as.character(test.y[i]))
}
rownames(test.pp)=rownames(test.p)
colnames(test.pp)=c("predict_probability","predict_group","true_group")
write.table(test.pp, pr.dir, sep = "\t", quote = F, col.names = NA)

# test ROC
plot_roc(test.y, test.p[, 2])
dev.off() 

#####################################importance
imp=train.rf$importance
write.table(imp,paste0(prefix,"_importance.txt"),quote=F,sep="\t",col.names=NA)
pdf(paste0(prefix,"_importance.pdf"),10,4)

par(mar=c(5,20,2,2)+0.01)
barplot(imp[,3],space=0.5,width=0.5,horiz=T,yaxt="n",xlab="Mean Decrease Accuracy",col="skyblue",border="skyblue")
axis(side=2,at=seq(0.5,nrow(imp[,3,drop=F])*0.75,0.75),labels=rownames(imp),las=1,cex.axis=0.7)

dev.off()

