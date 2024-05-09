library(pROC)

# function
plot_roc <- function(response, predictor, direction = "auto") {
  print(paste0("warning: levels of respone is ", paste(levels(as.factor(response)), collapse = ", "), " and should corresponding to controal and case, the default direction is auto"))
  roc.obj <- roc(response, predictor, percent = T, ci = T, plot = T, direction = direction)
  ci.se.obj <- ci.se(roc.obj, specificities = seq(0, 100, 5))
  plot(ci.se.obj, type = "shape", col = rgb(0, 1, 0, alpha = 0.2))
  plot(ci.se.obj, type = "bars")
  plot(roc.obj, col = 2, add = T)
  txt <- c(paste("AUC=", round(roc.obj$ci[2], 2), "%"), paste("95% CI:", round(roc.obj$ci[1], 2), "%-", round(roc.obj$ci[3], 2), 
    "%"))
  legend("bottomright", txt)
} 
