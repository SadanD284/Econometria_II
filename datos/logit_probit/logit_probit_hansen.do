/*===========================================================================
  
  Econometría II — Unidad 1: Modelo Logit y Modelo Probit
  Sadan De la Cruz ALmanza
  Departamento de Economía
  Universidad de Pamplona
  ---------------------------------------------------------------------------
  
  Replicación del Ejemplo de Hansen con datos CPS09
  Basado en: Hansen, B. (2022). Econometrics. Princeton University Press.
             Capítulo 25 — Modelos de Elección Discreta
  Datos: Current Population Survey (CPS) 2009 — Submuestra de hombres
         con educación universitaria.
  Archivo: cps09mar.dta
	
==========================================================================*/


**# Configuración general

clear all
set more off
capture log close
log using "logit_probit_hansen.log", replace text

* Cargar la base de datos
use "cps09mar.dta", clear

/*
  Variable dependiente: union (binaria)
  Ejercicio de Hansen: modelar la probabilidad de estar sindicalizado
  en función de características del trabajador.
*/


**# Construcción de variables

gen married   = (marital == 1)         // Casado con cónyuge presente
gen nonwhite  = (race > 1)             // No blanco
gen exp       = age - education - 6    // Experiencia potencial (Mincer)
gen exp2      = exp^2 / 100            // Experiencia al cuadrado (escalada)
gen lnearnings = ln(earnings)          // Log de ingresos (para análisis auxiliar)

gen south  = (region == 3)
gen midwest= (region == 2)
gen west   = (region == 4)

label variable union      "Miembro de sindicato (1=Sí)"
label variable female     "Mujer (1=Sí)"
label variable hisp       "Hispano (1=Sí)"
label variable married    "Casado con cónyuge presente"
label variable nonwhite   "No blanco"
label variable exp        "Experiencia potencial (años)"
label variable exp2       "Experiencia potencial² / 100"
label variable education  "Años de escolaridad"
label variable south      "Región Sur"
label variable midwest    "Región Medio Oeste"
label variable west       "Región Oeste"

/*
  Especificación del modelo:
  
  Seguimos la especificación de Hansen (2022, Cap. 15).
  La variable dependiente es:
        union_i ∈ {0, 1}
  El vector de regresores incluye:
        X_i = {female, hisp, education, exp, exp²,
               married, nonwhite, south, midwest, west}
  Estimamos: P(union_i = 1 | X_i) mediante tres enfoques:
        1. MPL  — Mínimos Cuadrados Ordinarios (referencia)
        2. Logit — Función logística acumulada (MV)
        3. Probit — Función normal estándar acumulada (MV)
*/


**# Estadísticas descriptivas

summarize union female hisp married nonwhite education exp ///
          south midwest west, separator(0)

tabulate union, missing

tabulate female union, row nofreq

tabulate region union, row nofreq

**# Modelo Lineal de Probabilidad (MLP)
  
regress union female hisp education exp exp2 married nonwhite ///
        south midwest west, robust

estimates store mlp

predict phat_mpl, xb
count if phat_mpl < 0 // valores predichos por debajo de 0
count if phat_mpl > 1 // valores predichos por arriba de 1

summarize phat_mpl, detail

**# Modelo Logit 
  
logit union female hisp education exp exp2 married nonwhite ///
      south midwest west, robust nolog

estimates store logit1

predict phat_logit, pr

summarize phat_logit, detail


logit union female hisp education exp exp2 married nonwhite ///
      south midwest west, robust nolog or // odds

/*
  Interpretación:
  ─────────────────────────────────────────────────────────────────────────
  Odds: veces que aumenta el "chance" de estar sindicalizado. El coeficiente
  solo se puede interpretar como un odds, no de manera directa y tampoco en este 
  caso como tradicionalmente lo hacemos en términos de los "efectos".
  
  si odds > 1 (aumenta el chance -> (odds - 1) * 100
  si odds < 1 (disminuye el chance) -> (1 - odds) * 100
  
  female = 0.8964 
  "Ser mujer está asociado con un chance de estar sindicalizado en 10.36% menor
  que los hombres (significativo marginalmente, en este caso)"
  
*/

**# Modelo Probit

probit union female hisp education exp exp2 married nonwhite ///
       south midwest west, robust nolog

estimates store probit1

predict phat_probit, pr

summarize phat_probit, detail

**#  Efectos marginales

/*
  Los coeficientes de Logit y Probit NO son efectos marginales directos.
  
  Dos enfoques estándar:
  (a) AME — Average Marginal Effect: promedio sobre toda la muestra
      EM = (1/n) Σ_i ∂P(Y_i=1|X_i)/∂X_k
  (b) MEM — Marginal Effect at the Mean: evaluado en X̄
*/


**## Efectos marginales promedio (AME)

logit union female hisp education exp exp2 married nonwhite ///
      south midwest west, robust nolog
margins, dydx(*) post
estimates store ame_logit

probit union female hisp education exp exp2 married nonwhite ///
       south midwest west, robust nolog
margins, dydx(*) post
estimates store ame_probit

/*
  Nota: las salidas de los coeficientes aparecen en proporciones inicialmente.
*/


**## Efectos marginales en la media (MEM)

probit union female hisp education exp exp2 married nonwhite ///
       south midwest west, robust nolog
margins, dydx(*) atmeans post
estimates store mem_probit

**# Tablas de comparación

ssc install estout, replace // instalar la opción de diseño de tabla
	
esttab mlp logit1 probit1, ///
    b(%9.4f) se(%9.4f) ///
    stats(N ll r2 r2_p, fmt(%9.0f %9.2f %9.4f %9.4f)) ///
    title("Tabla 1. Coeficientes: MPL, Logit y Probit") ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    mtitles("MPL" "Logit" "Probit")

**# Criterios de selección 
/*
  
  A diferencia de MCO, los modelos Logit/Probit no tienen un R² clásico.
  Se usan pseudo-R² y medidas de información:
    (a) Pseudo-R² de McFadden: 1 − (logL_completo / logL_restringido)
    (b) AIC = −2·logL + 2·k
    (c) BIC = −2·logL + k·ln(n)
    (d) Tasa de predicción correcta (matriz de clasificación)
    (e) Prueba de Hosmer-Lemeshow (calibración del modelo)

*/


quietly logit union female hisp education exp exp2 married nonwhite ///
      south midwest west, robust nolog
estat ic                         // AIC y BIC
estat classification             // Tabla de clasificación
lrtest, saving(0)                // Base para LR test

/*
  Matriz de confusión 
  
  NotebookLM: ¿Qué es una matriz de confusión?
  
        Predicho=0  Predicho=1
  Real=0  TN         FP
  Real=1  FN         TP
  Sensibilidad = TP/(TP+FN); Especificidad = TN/(TN+FP)
*/

quietly probit union female hisp education exp exp2 married nonwhite ///
       south midwest west, robust nolog
estat ic
estat classification

**# Curva ROC (Receiver Operating Characteristic)

/*
Se construye graficando la sensibilidad frente a la especificidad para todos
los posibles niveles de umbral. 


*/

**## ROC Logit
quietly logit union female hisp education exp exp2 married nonwhite ///
      south midwest west, robust nolog
predict roc_logit, pr
roctab union roc_logit, graph title("Curva ROC — Logit")

**## ROC Probit
quietly probit union female hisp education exp exp2 married nonwhite ///
       south midwest west, robust nolog
predict roc_probit, pr
roctab union roc_probit, graph title("Curva ROC — Probit")

/*
  Nota: El área bajo la curva ROC (AUC) mide discriminación:
  
  AUC = 0.5 → sin poder predictivo (igual que azar)
  AUC = 1.0 → predicción perfecta
  AUC > 0.8 → modelo con buen desempeño discriminatorio

*/

**# Pruebas de hipótesis

/*
  En modelos MV se utilizan tres tipos de pruebas equivalentes asintóticas:
    (a) Prueba de Razón de Verosimilitud (LR): −2(logL_R − logL_NR)
    (b) Prueba de Wald: (Rβ̂ − r)' [R V(β̂) R']⁻¹ (Rβ̂ − r)
    (c) Prueba del multiplicador de Lagrange (Score)
*/

* Significancia conjunta (prueba de Wald) 

quietly logit union female hisp education exp exp2 married nonwhite ///
      south midwest west, robust nolog
test south midwest west // controles regionales

quietly probit union female hisp education exp exp2 married nonwhite ///
       south midwest west, robust nolog
test south midwest west

* LR test: modelo con y sin controles de región (modelo completo vs constante)

quietly logit union female hisp education exp exp2 married nonwhite ///
      south midwest west, nolog
estimates store M_completo

quietly logit union female hisp education exp exp2 married nonwhite, nolog
estimates store M_sinregion

lrtest M_sinregion M_completo

* Prueba de cuadratura en exp (linealidad de experiencia) ───

quietly logit union female hisp education exp exp2 married nonwhite ///
      south midwest west, robust nolog
test exp exp2

* Tabla general de pruebas

fitstat 

**# Análisis de probabilidades predichas

/*
  Ejercicio de simulación: variación en la probabilidad de sindicalización
  ante cambios en las características del trabajador, manteniendo el resto
  en sus valores medios (ejercicio de ceteris paribus).
*/


* Probabilidad predicha por género, promediando el resto

quietly logit union female hisp education exp exp2 married nonwhite ///
      south midwest west, robust nolog
margins hisp, atmeans

* Perfil de probabilidad por años de experiencia 

quietly logit union female hisp education exp exp2 married nonwhite ///
      south midwest west, robust nolog
margins, at(exp=(0(5)40)) atmeans vsquish
marginsplot, title("Prob. de sindicalización vs. experiencia (Logit)") ///
    xtitle("Experiencia potencial (años)") ytitle("P(union=1)") ///
    recastci(rarea) ciopts(color(%30))

*Perfil de probabilidad por años de educación

quietly probit union female hisp education exp exp2 married nonwhite ///
       south midwest west, robust nolog
margins, at(education=(8(2)20)) atmeans vsquish
marginsplot, title("Prob. de sindicalización vs. educación (Probit)") ///
    xtitle("Años de educación") ytitle("P(union=1)") ///
    recastci(rarea) ciopts(color(%30))

log close
