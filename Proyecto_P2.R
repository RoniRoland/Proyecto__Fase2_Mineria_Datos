library(arules)
library(readxl)
library(ggplot2)
library(purrr)
library(dplyr)
library(rpart)
library(rpart.plot)
library(randomForest)

ruta <- "F:/Documentos/Docs Roni/Documentos Roni/Docs U AND Cursos/USAC MAESTRIA/MAESTRIA DATA ANLISIS/Cursos/Mineria de datos/Proyecto 2/data_hospitalaria/"

archivos <- list.files(ruta, pattern = "\\.xlsx$", full.names = TRUE)

leer_base <- function(path) {
  read_excel(path)
}

datos <- map_dfr(archivos, leer_base)

#============================ ARBOL 1 =====================================

datos_modelo <- datos %>%
  filter(
    DIASESTAN >= 1, DIASESTAN <= 98,
    EDAD >= 0, EDAD <= 99
  ) %>%
  mutate(
    TRATA_COMPLEJO = case_when(
      TRATARECIB == 1 ~ "SIMPLE",
      TRATARECIB %in% c(2, 3) ~ "COMPLEJO",
      TRUE ~ NA_character_
    )
  ) %>%
  filter(!is.na(TRATA_COMPLEJO))


table(datos_modelo$TRATA_COMPLEJO)


data_arbol1 <- datos_modelo %>%
  select(
    TRATA_COMPLEJO,
    EDAD,
    SEXO,
    PPERTENENCIA,
    DIASESTAN
  ) %>%
  mutate(
    TRATA_COMPLEJO = factor(TRATA_COMPLEJO),
    SEXO           = factor(SEXO),
    PPERTENENCIA   = factor(PPERTENENCIA)
  ) %>%
  na.omit()

arbol_tratamiento <- rpart(
  TRATA_COMPLEJO ~ EDAD + SEXO + PPERTENENCIA + DIASESTAN,
  data   = data_arbol1,
  method = "class",
  control = rpart.control(
    maxdepth = 4,     
    minbucket = 3000, 
    cp = 0.01         
  )
)

printcp(arbol_tratamiento)  

rpart.plot(
  arbol_tratamiento,
  type          = 2,
  extra         = 104,
  under         = TRUE,
  fallen.leaves = TRUE,
  box.palette   = "Blues",
  branch.lty    = 1,
  shadow.col    = "gray",
  varlen        = 0,
  faclen        = 0,
  cex           = 0.8,
  main          = "Árbol 1 – Predicción de Tratamiento (Simple vs Complejo)"
)

#============================ ESCENARIOS 1 ====================================


escenarioA <- data.frame(
  EDAD         = 22,
  SEXO         = factor(2, levels = levels(data_arbol1$SEXO)),
  PPERTENENCIA = factor(1, levels = levels(data_arbol1$PPERTENENCIA)),
  DIASESTAN    = 2
)

predict(arbol_tratamiento, escenarioA, type = "class")
predict(arbol_tratamiento, escenarioA, type = "prob")


esc2 <- data.frame(
  EDAD = 22,
  SEXO = factor(2, levels = levels(data_arbol1$SEXO)),
  PPERTENENCIA = factor(2, levels = levels(data_arbol1$PPERTENENCIA)),
  DIASESTAN = 1
)

predict(arbol_tratamiento, esc2, type = "class")
predict(arbol_tratamiento, esc2, type = "prob")


esc3 <- data.frame(
  EDAD = 10,
  SEXO = factor(2, levels = levels(data_arbol1$SEXO)),
  PPERTENENCIA = factor(3, levels = levels(data_arbol1$PPERTENENCIA)),
  DIASESTAN = 1
)

predict(arbol_tratamiento, esc3, type = "class")
predict(arbol_tratamiento, esc3, type = "prob")

esc4 <- data.frame(
  EDAD = 65,
  SEXO = factor(1, levels = levels(data_arbol1$SEXO)),
  PPERTENENCIA = factor(1, levels = levels(data_arbol1$PPERTENENCIA)),
  DIASESTAN = 4
)

predict(arbol_tratamiento, esc4, type = "class")
predict(arbol_tratamiento, esc4, type = "prob")


#============================ ARBOL 2 =====================================


datos_arbol2 <- datos %>%
  filter(
    DIASESTAN >= 1, DIASESTAN <= 98,
    EDAD >= 0, EDAD <= 99,
    MES >= 1, MES <= 12
  ) %>%
  mutate(
    MES = factor(MES, levels = 1:12,
                 labels = c("Enero","Febrero","Marzo","Abril","Mayo","Junio",
                            "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre")),
    SEXO          = factor(SEXO),
    TRATARECIB    = factor(TRATARECIB),
    PPERTENENCIA  = factor(PPERTENENCIA)
  ) %>%
  select(MES, EDAD, SEXO, TRATARECIB, DIASESTAN, PPERTENENCIA) %>%
  na.omit()

arbol_mes <- rpart(
  MES ~ EDAD + SEXO + TRATARECIB + DIASESTAN + PPERTENENCIA,
  data   = datos_arbol2,
  method = "class",
  control = rpart.control(
    cp        = 0.0001,   
    minsplit  = 3000,
    minbucket = 1200,
    maxdepth  = 6,
    xval      = 3
  )
)

printcp(arbol_mes)  

cp_simple <- 0.00057161  

arbol_mes_simple <- prune(arbol_mes, cp = cp_simple)

rpart.plot(
  arbol_mes_simple,
  type          = 2,      
  extra         = 2,    
  under         = TRUE,
  fallen.leaves = TRUE,
  box.palette   = "Purples",
  branch.lty    = 1,
  shadow.col    = "gray",
  varlen        = 0,     
  faclen        = 0,      
  cex           = 0.6,
  main          = "Árbol 2 – Predicción del Mes de Ingreso Hospitalario"
)


#============================ ESCENARIOS 2 ====================================



escenario1 <- data.frame(
  EDAD         = 35,
  SEXO         = factor(1,  levels = levels(datos_arbol2$SEXO)),      
  TRATARECIB   = factor(1,  levels = levels(datos_arbol2$TRATARECIB)),
  DIASESTAN    = 3,
  PPERTENENCIA = factor(2,  levels = levels(datos_arbol2$PPERTENENCIA))
)

predict(arbol_mes_simple, escenario1, type = "class")
predict(arbol_mes_simple, escenario1, type = "prob")



escenario2 <- data.frame(
  EDAD         = 50,
  SEXO         = factor(1,  levels = levels(datos_arbol2$SEXO)),   
  TRATARECIB   = factor(1,  levels = levels(datos_arbol2$TRATARECIB)),
  DIASESTAN    = 1,
  PPERTENENCIA = factor(4,  levels = levels(datos_arbol2$PPERTENENCIA))
)

predict(arbol_mes_simple, escenario2, type = "class")
predict(arbol_mes_simple, escenario2, type = "prob")


escenario3 <- data.frame(
  EDAD         = 70,
  SEXO         = factor(2,  levels = levels(datos_arbol2$SEXO)),        
  TRATARECIB   = factor(2,  levels = levels(datos_arbol2$TRATARECIB)),
  DIASESTAN    = 7,
  PPERTENENCIA = factor(4,  levels = levels(datos_arbol2$PPERTENENCIA))
)

predict(arbol_mes_simple, escenario3, type = "class")
predict(arbol_mes_simple, escenario3, type = "prob")


escenario4 <- data.frame(
  EDAD         = 30,
  SEXO         = factor(2,  levels = levels(datos_arbol2$SEXO)),      
  TRATARECIB   = factor(1,  levels = levels(datos_arbol2$TRATARECIB)),
  DIASESTAN    = 3,
  PPERTENENCIA = factor(4,  levels = levels(datos_arbol2$PPERTENENCIA))
)

predict(arbol_mes_simple, escenario4, type = "class")
predict(arbol_mes_simple, escenario4, type = "prob")


#============================ ARBOL 3 =====================================


datos_modelo3 <- datos %>%
  filter(
    DIASESTAN >= 1, DIASESTAN <= 98,
    EDAD >= 0, EDAD <= 99
  ) %>%
  mutate(
    EDAD_RANGO = case_when(
      EDAD < 15 ~ "NIÑO",
      EDAD < 40 ~ "ADULTO_JOVEN",
      EDAD < 65 ~ "ADULTO",
      TRUE ~ "ADULTO_MAYOR"
    ),
    EDAD_RANGO = factor(EDAD_RANGO)
  ) %>%
  select(EDAD_RANGO, SEXO, PPERTENENCIA, TRATARECIB, DIASESTAN) %>%
  mutate(
    SEXO = factor(SEXO),
    PPERTENENCIA = factor(PPERTENENCIA),
    TRATARECIB = factor(TRATARECIB)
  ) %>%
  na.omit()


arbol_edad <- rpart(
  EDAD_RANGO ~ SEXO + PPERTENENCIA + TRATARECIB + DIASESTAN,
  data = datos_modelo3,
  method = "class",
  control = rpart.control(
    maxdepth = 5,
    minbucket = 2000,
    cp = 0.001
  )
)

printcp(arbol_edad)

rpart.plot(
  arbol_edad,
  type = 2,
  extra = 2,       
  under = TRUE,
  fallen.leaves = TRUE,
  box.palette = "Purples",
  branch.lty = 1,
  shadow.col = "gray",
  varlen = 0,
  faclen = 0,
  cex = 0.7,
  main = "Árbol 3 – Predicción del Rango de Edad"
)


#============================ ESCENARIOS 3 ====================================


esc_nino <- data.frame(
  SEXO = factor(2, levels = levels(datos_modelo3$SEXO)),
  PPERTENENCIA = factor(1, levels = levels(datos_modelo3$PPERTENENCIA)),
  TRATARECIB = factor(2, levels = levels(datos_modelo3$TRATARECIB)),
  DIASESTAN = 4
)

predict(arbol_edad, esc_nino, type = "class")
predict(arbol_edad, esc_nino, type = "prob")

esc_adulto_joven <- data.frame(
  SEXO = factor(1, levels = levels(datos_modelo3$SEXO)),
  PPERTENENCIA = factor(4, levels = levels(datos_modelo3$PPERTENENCIA)),
  TRATARECIB = factor(2, levels = levels(datos_modelo3$TRATARECIB)),
  DIASESTAN = 1
)

predict(arbol_edad, esc_adulto_joven, type = "class")
predict(arbol_edad, esc_adulto_joven, type = "prob")


esc_adulto <- data.frame(
  SEXO = factor(1, levels = levels(datos_modelo3$SEXO)),
  PPERTENENCIA = factor(4, levels = levels(datos_modelo3$PPERTENENCIA)),
  TRATARECIB = factor(3, levels = levels(datos_modelo3$TRATARECIB)),
  DIASESTAN = 2
)

predict(arbol_edad, esc_adulto, type = "class")
predict(arbol_edad, esc_adulto, type = "prob")


esc_adulto_mayor <- data.frame(
  SEXO = factor(2, levels = levels(datos_modelo3$SEXO)),
  PPERTENENCIA = factor(2, levels = levels(datos_modelo3$PPERTENENCIA)),
  TRATARECIB = factor(2, levels = levels(datos_modelo3$TRATARECIB)),
  DIASESTAN = 3
)

predict(arbol_edad, esc_adulto_mayor, type = "class")
predict(arbol_edad, esc_adulto_mayor, type = "prob")


#============================ ARBOL 4 =====================================


datos_arbol4 <- datos %>%
  filter(
    DIASESTAN >= 1, DIASESTAN <= 98,
    EDAD >= 0, EDAD <= 99,
    MES >= 1, MES <= 12
  ) %>%
  mutate(
    TRATA_TIPO    = factor(TRATARECIB),   
    SEXO          = factor(SEXO),
    PPERTENENCIA  = factor(PPERTENENCIA),
    MES           = factor(MES)
  ) %>%
  select(
    TRATA_TIPO,  
    EDAD,
    DIASESTAN,
    SEXO,
    PPERTENENCIA,
    MES
  ) %>%
  na.omit()

table(datos_arbol4$TRATA_TIPO)
prop.table(table(datos_arbol4$TRATA_TIPO))


arbol_tratatipo <- rpart(
  TRATA_TIPO ~ EDAD + DIASESTAN + SEXO + PPERTENENCIA + MES,
  data   = datos_arbol4,
  method = "class",
  control = rpart.control(
    maxdepth  = 5,     
    minbucket = 3000,   
    cp        = 0.001 
  )
)

printcp(arbol_tratatipo)  

rpart.plot(
  arbol_tratatipo,
  type          = 2,
  extra         = 2,     
  under         = TRUE,
  fallen.leaves = TRUE,
  box.palette   = "Oranges",
  branch.lty    = 1,
  shadow.col    = "gray",
  varlen        = 0,
  faclen        = 0,
  cex           = 0.7,
  main          = "Árbol 4 – Predicción del Tipo de Tratamiento"
)


#============================ ESCENARIOS 4 ====================================


esc4_a <- data.frame(
  SEXO = factor(1, levels = levels(datos_arbol4$SEXO)),
  EDAD = 55,
  DIASESTAN = 1,
  PPERTENENCIA = factor(4, levels = levels(datos_arbol4$PPERTENENCIA)),
  MES = factor(5, levels = levels(datos_arbol4$MES))
)

predict(arbol_tratatipo, esc4_a, type = "class")
predict(arbol_tratatipo, esc4_a, type = "prob")


esc4_b <- data.frame(
  SEXO = factor(2, levels = levels(datos_arbol4$SEXO)),
  EDAD = 7,
  DIASESTAN = 1,
  PPERTENENCIA = factor(1, levels = levels(datos_arbol4$PPERTENENCIA)),
  MES = factor(3, levels = levels(datos_arbol4$MES))
)

predict(arbol_tratatipo, esc4_b, type = "class")
predict(arbol_tratatipo, esc4_b, type = "prob")


esc4_c <- data.frame(
  SEXO = factor(1, levels = levels(datos_arbol4$SEXO)),
  EDAD = 10,
  DIASESTAN = 5,
  PPERTENENCIA = factor(3, levels = levels(datos_arbol4$PPERTENENCIA)),
  MES = factor(9, levels = levels(datos_arbol4$MES))
)

predict(arbol_tratatipo, esc4_c, type = "class")
predict(arbol_tratatipo, esc4_c, type = "prob")


esc4_d <- data.frame(
  SEXO = factor(2, levels = levels(datos_arbol4$SEXO)),
  EDAD = 28,
  DIASESTAN = 2,
  PPERTENENCIA = factor(2, levels = levels(datos_arbol4$PPERTENENCIA)),
  MES = factor(1, levels = levels(datos_arbol4$MES))
)

predict(arbol_tratatipo, esc4_d, type = "class")
predict(arbol_tratatipo, esc4_d, type = "prob")



#============================ RANDOM FOREST 1 ==================================


datos_rf1 <- datos %>%
  filter(
    DIASESTAN >= 1, DIASESTAN <= 98,
    EDAD >= 0, EDAD <= 99
  ) %>%
  mutate(
    TRATA_TIPO = case_when(
      TRATARECIB == 1 ~ "SIMPLE",
      TRATARECIB %in% c(2,3) ~ "COMPLEJO",
      TRUE ~ NA_character_
    ),
    TRATA_TIPO = factor(TRATA_TIPO),
    SEXO = factor(SEXO),
    PPERTENENCIA = factor(PPERTENENCIA)
  ) %>%
  select(TRATA_TIPO, EDAD, SEXO, DIASESTAN, PPERTENENCIA) %>%
  na.omit()

set.seed(100)
datos_rf1 <- datos_rf1[sample(1:nrow(datos_rf1)),]

index <- sample(1:nrow(datos_rf1), 0.8*nrow(datos_rf1))

train1 <- datos_rf1[index,]
test1  <- datos_rf1[-index,]

bosque1 <- randomForest(
  TRATA_TIPO ~ .,
  data = train1,
  ntree = 200,
  mtry = 3
)

pred1 <- predict(bosque1, test1)

matriz1 <- table(test1$TRATA_TIPO, pred1)
matriz1

precision1 <- sum(diag(matriz1)) / sum(matriz1)
precision1

plot(bosque1, main = "RF 1 – Error del Modelo (Tipo de Tratamiento)")


#============================ ESCENARIOS RF 1 ==================================


esc_rf1_A <- data.frame(
  EDAD         = 22,
  SEXO         = factor(2, levels = levels(train1$SEXO)),
  DIASESTAN    = 1,
  PPERTENENCIA = factor(1, levels = levels(train1$PPERTENENCIA))
)

predict(bosque1, esc_rf1_A, type = "class")
predict(bosque1, esc_rf1_A, type = "prob")


esc_rf1_B <- data.frame(
  EDAD         = 35,
  SEXO         = factor(1, levels = levels(train1$SEXO)),
  DIASESTAN    = 3,
  PPERTENENCIA = factor(4, levels = levels(train1$PPERTENENCIA))
)

predict(bosque1, esc_rf1_B, type = "class")
predict(bosque1, esc_rf1_B, type = "prob")

esc_rf1_C <- data.frame(
  EDAD         = 10,
  SEXO         = factor(2, levels = levels(train1$SEXO)),
  DIASESTAN    = 1,
  PPERTENENCIA = factor(3, levels = levels(train1$PPERTENENCIA))
)

predict(bosque1, esc_rf1_C, type = "class")
predict(bosque1, esc_rf1_C, type = "prob")



#============================ RANDOM FOREST 2 ==================================



datos_rf2 <- datos %>%
  filter(
    EDAD >= 0, EDAD <= 99,
    MES >= 1, MES <= 12
  ) %>%
  mutate(
    MES = factor(MES),
    SEXO = factor(SEXO),
    PPERTENENCIA = factor(PPERTENENCIA),
    TRATARECIB = factor(TRATARECIB)
  ) %>%
  select(MES, EDAD, DIASESTAN, SEXO, PPERTENENCIA, TRATARECIB) %>%
  na.omit()

set.seed(100)
datos_rf2 <- datos_rf2[sample(1:nrow(datos_rf2)),]

index <- sample(1:nrow(datos_rf2), 0.8*nrow(datos_rf2))

train2 <- datos_rf2[index,]
test2  <- datos_rf2[-index,]

bosque2 <- randomForest(
  MES ~ .,
  data = train2,
  ntree = 300,
  mtry = 3
)

pred2 <- predict(bosque2, test2)

matriz2 <- table(test2$MES, pred2)
matriz2

precision2 <- sum(diag(matriz2)) / sum(matriz2)
precision2

plot(bosque2, main = "RF 2 – Error del Modelo (Mes de Ingreso)")

#============================ ESCENARIOS RF 2 ==================================


esc_rf2_A <- data.frame(
  EDAD         = 22,
  SEXO         = factor(2, levels = levels(train2$SEXO)),
  DIASESTAN    = 1,
  TRATARECIB   = factor(1, levels = levels(train2$TRATARECIB)),
  PPERTENENCIA = factor(1, levels = levels(train2$PPERTENENCIA))
)

predict(bosque2, esc_rf2_A, type = "class")
predict(bosque2, esc_rf2_A, type = "prob")



esc_rf2_B <- data.frame(
  EDAD         = 70,
  SEXO         = factor(1, levels = levels(train2$SEXO)),
  DIASESTAN    = 10,
  TRATARECIB   = factor(3, levels = levels(train2$TRATARECIB)),
  PPERTENENCIA = factor(4, levels = levels(train2$PPERTENENCIA))
)

predict(bosque2, esc_rf2_B, type = "class")
predict(bosque2, esc_rf2_B, type = "prob")



#============================ RANDOM FOREST 3 ==================================

datos_rf3 <- datos %>%
  filter(EDAD >= 0, EDAD <= 99) %>%
  mutate(
    EDAD_RANGO = case_when(
      EDAD < 15 ~ "NIÑO",
      EDAD < 40 ~ "ADULTO_JOVEN",
      EDAD < 65 ~ "ADULTO",
      TRUE ~ "ADULTO_MAYOR"
    ),
    EDAD_RANGO = factor(EDAD_RANGO),
    SEXO = factor(SEXO),
    PPERTENENCIA = factor(PPERTENENCIA),
    TRATARECIB = factor(TRATARECIB)
  ) %>%
  select(EDAD_RANGO, DIASESTAN, SEXO, PPERTENENCIA, TRATARECIB) %>%
  na.omit()

set.seed(100)
datos_rf3 <- datos_rf3[sample(1:nrow(datos_rf3)),]

index <- sample(1:nrow(datos_rf3), 0.8*nrow(datos_rf3))

train3 <- datos_rf3[index,]
test3  <- datos_rf3[-index,]

bosque3 <- randomForest(
  EDAD_RANGO ~ .,
  data = train3,
  ntree = 250,
  mtry = 2
)

pred3 <- predict(bosque3, test3)

matriz3 <- table(test3$EDAD_RANGO, pred3)
matriz3

precision3 <- sum(diag(matriz3)) / sum(matriz3)
precision3

plot(bosque3, main = "RF 3 – Error del Modelo (Rango de Edad)")

#============================ ESCENARIOS RF 3 ==================================

esc_rf3_A <- data.frame(
  DIASESTAN    = 2,
  SEXO         = factor("2", levels = levels(train3$SEXO)),         
  PPERTENENCIA = factor("1", levels = levels(train3$PPERTENENCIA)), 
  TRATARECIB   = factor("1", levels = levels(train3$TRATARECIB))  
)

pred_rf3_A_class <- predict(bosque3, esc_rf3_A, type = "class")
pred_rf3_A_class

pred_rf3_A_prob  <- predict(bosque3, esc_rf3_A, type = "prob")
pred_rf3_A_prob



esc_rf3_B <- data.frame(
  DIASESTAN    = 10,
  SEXO         = factor("1", levels = levels(train3$SEXO)),          
  PPERTENENCIA = factor("4", levels = levels(train3$PPERTENENCIA)),  
  TRATARECIB   = factor("3", levels = levels(train3$TRATARECIB))  
)

pred_rf3_B_class <- predict(bosque3, esc_rf3_B, type = "class")
pred_rf3_B_class

pred_rf3_B_prob  <- predict(bosque3, esc_rf3_B, type = "prob")
pred_rf3_B_prob











