library(sparklyr)
library(dplyr)

sc <- spark_connect(master = "local", app_name = "ZETA Project - DS")
