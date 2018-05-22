# read data
SLM <- read.csv(file="/Users/vivienzou/Desktop/neurohackathon2018/Hackathon dataset 2018 - BARTH/SLM Distal apical dendrites/WT_SLM_distal_apical.csv", header=TRUE, sep=",")
SP_PO_A <- read.csv(file="/Users/vivienzou/Desktop/neurohackathon2018/Hackathon dataset 2018 - BARTH/SP-SO Proximal apical dendrites/WT_SP-SO_proximal_apical.csv", header=TRUE, sep=",")
SP_PO_B <- read.csv(file="/Users/vivienzou/Desktop/neurohackathon2018/Hackathon dataset 2018 - BARTH/SP-SO Proximal basal dendrites/WT_SP-SO_proximal_basal.csv", header=TRUE, sep=",")
SR <- read.csv(file="/Users/vivienzou/Desktop/neurohackathon2018/Hackathon dataset 2018 - BARTH/SR Medial apical dendrites/WT_SR_medial_apical.csv", header=TRUE, sep=",")
dev <- read.csv(file="/Users/vivienzou/Desktop/neurohackathon2018/total_dev.csv", header=TRUE, sep=",")

SLM_AD <- read.csv(file="/Users/vivienzou/Desktop/neurohackathon2018/Hackathon dataset 2018 - BARTH/SLM Distal apical dendrites/AD_SLM_distal_apical.csv", header=TRUE, sep=",")
SP_PO_B_AD <- read.csv(file="/Users/vivienzou/Desktop/neurohackathon2018/Hackathon dataset 2018 - BARTH/SP-SO Proximal basal dendrites/AD_SP-SO_proximal_basal.csv", header=TRUE, sep=",")
SP_PO_A_AD <- read.csv(file="/Users/vivienzou/Desktop/neurohackathon2018/Hackathon dataset 2018 - BARTH/SP-SO Proximal apical dendrites/AD_SP-SO_proximal_apical.csv", header=TRUE, sep=",")

print(head(SLM))

# ANOVA with intensity mean
Intensity.Mean.aov <- aov(Intensity.Mean ~ Area+Distance.from.Origin+Distance.to.Image.Border.XY+Ellipticity..oblate.+Ellipticity..prolate.
                +Number.of.Vertices+Position.X+Position.Y+Position.Z+Sphericity+Volume+Area*Number.of.Vertices
               +Area*Volume+Distance.from.Origin*Intensity.Min+Distance.from.Origin*Intensity.Mean+
                 Distance.from.Origin*Intensity.Median+Distance.to.Image.Border.XY*Intensity.Min+
                 Distance.to.Image.Border.XY*Intensity.Mean+Distance.to.Image.Border.XY*Intensity.Median+
                 Number.of.Vertices*Sphericity+Number.of.Vertices*Volume+Area*Sphericity+Sphericity*Volume
               ,
                data = SLM)
summary(Intensity.Mean.aov)

#ANOVA with Area
area.aov <- aov( Area ~ Intensity.Sum+Intensity.StdDev+Intensity.Max+Intensity.Min+Intensity.Mean+Distance.from.Origin+Distance.to.Image.Border.XY+Ellipticity..oblate.+Ellipticity..prolate.
                          +Number.of.Vertices+Position.X+Position.Y+Position.Z+Sphericity+Volume
                          #+Area*Number.of.Vertices
                          #+Area*Volume+Distance.from.Origin*Intensity.Min+Distance.from.Origin*Intensity.Mean+
                            #Distance.from.Origin*Intensity.Median+Distance.to.Image.Border.XY*Intensity.Min+
                            #Distance.to.Image.Border.XY*Intensity.Mean+Distance.to.Image.Border.XY*Intensity.Median+
                            #Number.of.Vertices*Sphericity+Number.of.Vertices*Volume+Area*Sphericity+Sphericity*Volume
                          ,
                          data = SLM)
summary(area.aov)

# correlation plot
library(corrplot)
corr_SLM<-data.frame(SLM$Area,SLM$Ellipticity..oblate.,SLM$Ellipticity..prolate,SLM$Intensity.Max,SLM$Intensity.Mean,SLM$Intensity.Median
       ,SLM$Intensity.Min,SLM$Intensity.StdDev,SLM$Intensity.Sum,SLM$Number.of.Vertices,SLM$Sphericity,SLM$Volume)

corr_SP_B<-data.frame(SP_PO_B$Area,SP_PO_B$Ellipticity..oblate.,SP_PO_B$Ellipticity..prolate,SP_PO_B$Intensity.Max,SP_PO_B$Intensity.Mean,SP_PO_B$Intensity.Median
                     ,SP_PO_B$Intensity.Min,SP_PO_B$Intensity.StdDev,SP_PO_B$Intensity.Sum,SP_PO_B$Number.of.Vertices,SP_PO_B$Sphericity,SP_PO_B$Volume)

corr_SLM_AD<-data.frame(SLM_AD$Area,SLM_AD$Ellipticity..oblate.,SLM_AD$Ellipticity..prolate,SLM_AD$Intensity.Max,SLM_AD$Intensity.Mean,SLM_AD$Intensity.Median
                     ,SLM_AD$Intensity.Min,SLM_AD$Intensity.StdDev,SLM_AD$Intensity.Sum,SLM_AD$Number.of.Vertices,SLM_AD$Sphericity,SLM_AD$Volume)


SLM_plot<-cor(corr_SLM)
SLM_AD_plot<-cor(corr_SLM_AD)
SP_PO_A_plot<-cor(SP_PO_A)
SP_PO_B_plot<-cor(corr_SP_B)
SR_plot<-cor(SR)
dev_plot<-cor(dev)
corrplot(SLM_plot, method="color")
corrplot(SLM_AD_plot, method="color")
corrplot(SP_PO_A_plot, method="color")
corrplot(SP_PO_B_plot, method="color")
corrplot(SR_plot, method="color")
corrplot(dev_plot,method="color")

#visualization
install.packages("ggpubr")

# pull out the first 10 data 
dplyr::sample_n(SLM, 10)


# SVM
library(e1071)
# Fitting model
SLM_SVM <-svm( Intensity.Mean ~ ., data = SLM)
summary(fit)
#Predict Output 
predicted= predict(fit,x_test)
# naive bayes
fit <-naiveBayes(Intensity.Mean ~ ., data = SLM)
summary(fit)
predicted= predict(fit,SLM$Area)
print(predicted)

# knn
library(kknn)
fit_knn <-knn(Intensity.Mean ~ ., data = SLM$Area, k=5)
summary(fit_knn)
predicted= predict(fit_knn,x_test)
# histogram 

hist(log(SLM$Area))
hist(log(SR$Area))
hist(log(SP_PO_A$Area))
hist(log(SP_PO_B$Area))

hist(log(SLM_AD$Area))

#two histogram
densAD <- density(SP_PO_A_AD$Volume)
densWT <- density(SP_PO_A$Volume)
xlim <- range(densWT$x,densAD$x)
ylim <- range(0,densWT$y, densAD$y)
ADCol <- rgb(1,0,0,0.2)
WTCol <- rgb(0,0,1,0.2)
plot(densAD, xlim = xlim, ylim = ylim, xlab = 'Lengths',
     main = 'Distribution of AD and WT', 
     panel.first = grid())
polygon(densAD, density = -1, col = ADCol)
polygon(densWT, density = -1, col = WTCol)


# k means
library(cluster)
k_means_SLM <- kmeans(SLM$Area, 20) 
k_means_SR <- kmeans(SR, 20) 
k_means_A <- kmeans(SP_PO_A, 20) 
k_means_B <- kmeans(SP_PO_B, 20) 
k_means_dev <- kmeans(dev$Area,3)
print(k_means_SLM)
print(k_means_A)
print(k_means_B)
print(k_means_SR)
print(k_means_dev)

dissE <- daisy(dev) 
dE2   <- dissE^2
sk2   <- silhouette(k_means_dev$cluster,dE2)
plot(sk2)

library(fpc)
plotcluster(dev$Volumn, k_means_dev$cluster)

#Dimensionality reduction
library(stats)
pca <- princomp(dev, cor = TRUE)
train  <- predict(pca,dev)
train_pca <- prcomp(dev, center = TRUE, scale = TRUE) 
print(train)
summary(train_pca)
biplot(train_pca, xlabs = rep("", nrow(train)))
summary(train)

# ABC
library(ABCoptim)
k <- 4
n <- 40
#model_AD <- data.frame(SLM_AD$Area,SLM_AD$Ellipticity..oblate.,SLM_AD$Sphericity,SLM_AD$Volume)
#model <- data.frame(SLM$Area,SLM$Ellipticity..oblate.,SLM$Sphericity,SLM$Volume)
w <-  matrix(SLM_AD$Intensity.Mean, ncol = 1, nrow = 40)    # This are the model parameters
X <- matrix(dev$Area, ncol = k, nrow = n) # This are the controls
y <- X %*% w   
# Objective function
fun <- function(x) {
  sum((y - X%*%x)^2)
}
ans <- abc_optim(rep(0,k), fun, lb = -10000, ub=10000)
coef(lm(y~0+X))
