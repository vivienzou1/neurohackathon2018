SLM <- read.csv(file="/Users/vivienzou/Desktop/2018neurohackathan/Hackathon dataset 2018 - BARTH/SLM Distal apical dendrites/WT_SLM_distal_apical.csv", header=TRUE, sep=",")
SP_PO_A <- read.csv(file="/Users/vivienzou/Desktop/2018neurohackathan/Hackathon dataset 2018 - BARTH/SP-SO Proximal apical dendrites/WT_SP-SO_proximal_apical.csv", header=TRUE, sep=",")
SP_PO_B <- read.csv(file="/Users/vivienzou/Desktop/2018neurohackathan/Hackathon dataset 2018 - BARTH/SP-SO Proximal basal dendrites/WT_SP-SO_proximal_basal.csv", header=TRUE, sep=",")
SR <- read.csv(file="/Users/vivienzou/Desktop/2018neurohackathan/Hackathon dataset 2018 - BARTH/SR Medial apical dendrites/WT_SR_medial_apical.csv", header=TRUE, sep=",")

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
SLM_plot<-cor(SLM)
SP_PO_A_plot<-cor(SP_PO_A)
SP_PO_B_plot<-cor(SP_PO_B)
SR_plot<-cor(SR)
corrplot(SLM_plot, method="color")
corrplot(SP_PO_A_plot, method="color")
corrplot(SP_PO_B_plot, method="color")
corrplot(SR_plot, method="color")

#visualization
install.packages("ggpubr")
# see the data 
dplyr::sample_n(SLM, 10)