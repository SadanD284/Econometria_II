/*===========================================================================
  
  Econometría II — Unidad 1: Modelo Lineal de Probabilidad (MPL)
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
version 16

global datos "/Users/sadan/Documents/GitHub/Econometria_II/datos/mlp"  // ruta directorio principal de los datos (cambia según cada estudiante)
global output "/Users/sadan/Documents/GitHub/Econometria_II/datos/mlp"         // Directorio para guardar gráficos y resultados (cambia según cada estudiante)

* Cargar base de datos
use "${datos}/cps09mar.dta", clear

* Nota: los global son de referencia, el estudiante puede abrir el archivo sin necesidad
*       de hacer este proceso.



describe 

/*-------------------------------------------------------------------------
  La base CPS09 incluye la variable "marital" como una variable numérica
  que codifica el estado civil. Crearemos una indicadora de casado.
  
  Convención CPS — variable marital (según codificación CPS/Hansen):
    1 = Married, spouse present
    2 = Married, spouse absent
    3 = Separated
    4 = Divorced
    5 = Widowed
    6 = Never married
-------------------------------------------------------------------------*/

* - Etiquetado de variables categóricas 

    label define civil ///
        1 "Casado, cónyuge presente" ///
        2 "Casado, cónyuge ausente"  ///
        3 "Separado"                 ///
        4 "Divorciado"               ///
        5 "Viudo"                    ///
        6 "Soltero/nunca casado"	 ///
		7 "Nunca casado"
    label values marital civil

tab marital
tab marital, nolab // sin etiqueta (modo de verificación)

generate byte married = (marital == 1) if !missing(marital) // variable binaria (Y_i)

* — Etiqueta de variable 

label variable married   "Casado con cónyuge presente (1=Sí, 0=No)"
label variable age       "Edad en años cumplidos"
label variable female    "Sexo femenino (1=Mujer, 0=Hombre)"
label variable education "Años de educación"
label variable earnings  "Ingreso semanal en USD"
label variable hours     "Horas trabajadas por semana"

label define yesno 0 "No" 1 "Sí" // define un "libro" de etiquetas para replicar 

label values married yesno
label values female  yesno

codebook married
tab married, missing


**## 1.1 Operaciones con variables binarias

summarize married

tab married female, row col chi2

* Crear otras variables binarias de interés
generate byte college = (education >= 16) if !missing(education)
label variable college "Educación universitaria completa (≥16 años educ.)"
label values college yesno

* Variable binaria a partir de condición continua
generate byte age40plus = (age >= 40) if !missing(age)
label variable age40plus "Edad 40 años o más"
label values age40plus yesno


**## 1.2 - Preparación (submuestra)

* En el ejemplo de Econometría de Hansen, el autor crea una submuestra haciendo
* las siguientes restricciones:
*    - Hombres (female == 0)
*    - Educación universitaria (education >= 16)
*    - Empleados con horas positivas

* Nota: consultar en el cuaderno de notebooklm del curso, los motivos por los cuales
* Hansen hace dicha restricción.

keep if female    == 0
keep if education >= 16
keep if hours     >  0
keep if earnings  >  0

count // conteo de observaciones

bysort age: summarize married, meanonly
preserve
    collapse (mean) p_married=married (count) n=married, by(age)
    label variable p_married "P(casado | edad)"
    label variable n         "Observaciones por edad"
    list if age >= 25 & age <= 65, sep(0)
restore

**# 2 - Estadísticas descriptivas

summarize age married education earnings hours, detail

tab married

by married, sort: summarize age

**# 3 - Estamación del MLP

/*-------------------------------------------------------------------------
 
  Este es el modelo que Hansen estima para ilustrar el MPL:
  
    P(married_i = 1 | age_i) = β₀ + β₁·age_i + u_i
	
-------------------------------------------------------------------------*/

* MPL sin corrección de heterocedasticidad (MCO clásico)

regress married age

scalar b0_mpl = _b[_cons]
scalar b1_mpl = _b[age]

* MPL con errores estándar robustos (corrección White — recomendada)

regress married age, robust
estimates store mpl_robusto

**# 4 - Diagnóstico de los problemas de un MLP

quietly regress married age, robust

**## 4.1 - No normalidad de u_i

/*-------------------------------------------------------------------------

  En el MPL, u_i solo puede tomar dos valores:
    u_i = 1 − (β₀ + β₁·age_i)  cuando Y_i = 1
    u_i = 0 − (β₀ + β₁·age_i)  cuando Y_i = 0
  Esto viola el supuesto de normalidad de los errores.
-------------------------------------------------------------------------*/

predict double resid_mpl, residuals

* Nota: Si el MLP es adecuado, los residuos no deberían rechazar normalidad, 
* un p-valor < 0.05 confirma la patología de no normalidad.

sktest resid_mpl    				// Jarque-Bera
swilk  resid_mpl if _n <= 2000     // Shapiro-Wilk válido hasta n=5000

histogram resid_mpl, bin(40) normal ///
    title("Residuos MPL — Distribución bimodal (no normal)") ///
    xtitle("Residuos") ytitle("Densidad") ///
    note("Se espera distribución bimodal: viola supuesto de normalidad") ///
    scheme(s1color)

**## 4.2 - Heterocedasticidad inherente
	
/*-------------------------------------------------------------------------

  Var(u_i | X_i) = P_i(1 − P_i) = (β₀ + β₁·X_i)(1 − β₀ − β₁·X_i)
  La varianza del error cambia con X → heterocedasticidad por diseño.
-------------------------------------------------------------------------*/

quietly regress married age
estat hettest // Prueba de Breusch-Pagan

estat imtest, white // Prueba de White

* Nota: en ambas pruebas deberían rechazar homocedasticidad (p < 0.05), 
* 		esto confirma que MCO deja de ser el MELI en el MLP.


**## 4.3 - Probabilidad fuera del rango 

predict double yhat_mpl, xb

summarize yhat_mpl

count if yhat_mpl < 0
count if yhat_mpl > 1

* Nota: esto corresponde a "predicciones absurdas" según Hansen

**### 4.3.1 - Predicciones fuera de rango 

di _newline "Coeficientes estimados del MPL:"
di "  β₀ (intercepto) = " b0_mpl
di "  β₁ (edad)       = " b1_mpl

* Calcular predicción para un hombre de 80 años
scalar edad_80   = 80
scalar prob_80   = b0_mpl + b1_mpl * edad_80

di "P(casado | edad=80) = " b0_mpl " + " b1_mpl " × 80"
di "P(casado | edad=80) = " prob_80
di "(Si este valor supera 1.00, reproduce el 'absurdo' de Hansen: >100%)"

* Tabla de predicciones para edades seleccionadas

foreach a in 25 30 35 40 45 50 55 60 65 70 75 80 {
    scalar prob_a = b0_mpl + b1_mpl * `a'
    local valida = cond(prob_a >= 0 & prob_a <= 1, "Sí [0,1]", "NO — fuera de rango")
    di %7.0f `a' " | " %20.4f prob_a " | `valida'"
}

* Nota: Hansen señala que para la edad de 80 años, el MLP predice una probabilidades
* superior al 100%, lo cual no tiene "sentido" económico. 

**# 5 - Visualización

**# 5.1 - proporción observada de casados por edad (replica Figura 25.1)

preserve

    collapse (mean) p_obs=married (count) n_obs=married, by(age)
    label variable p_obs "Proporción observada de casados"

    generate double p_mpl = b0_mpl + b1_mpl * age
    label variable p_mpl "Ajuste MPL: β₀ + β₁·edad"

    * Líneas de referencia para los límites de probabilidad
    generate limite_sup = 1
    generate limite_inf = 0

    /*---------------------------------------------------------------------
       Replica Figura 25.1b de Hansen, muestra:
        • Puntos: probabilidades observadas por edad (datos reales)
        • Línea roja: ajuste lineal del MPL
        • Líneas punteadas: límites de probabilidad válida [0, 1]
        • Las predicciones que "salen" de [0,1] hacen visible el absurdo
    ---------------------------------------------------------------------*/
    twoway ///
        (scatter p_obs age if age >= 20 & age <= 75, ///
            mcolor(navy%60) msize(small) msymbol(circle) ///
            mlabel("") ///
        ) ///
        (line p_mpl age if age >= 20 & age <= 85, ///
            lcolor(red) lwidth(medthick) lpattern(solid) ///
        ) ///
        (line limite_sup age, ///
            lcolor(black) lwidth(thin) lpattern(dash) ///
        ) ///
        (line limite_inf age, ///
            lcolor(black) lwidth(thin) lpattern(dash) ///
        ) ///
        , ///
        title("MLP — replicación Hansen (Fig. 25.1b)", ///
              size(medsmall)) ///
        subtitle("") ///
        xtitle("Edad") ///
        ytitle("P(casado)") ///
        xlabel(20(5)85) ///
        ylabel(0(0.1)1.2, format(%4.1f)) ///
        yline(0, lcolor(black) lwidth(thin) lpattern(dash)) ///
        yline(1, lcolor(black) lwidth(thin) lpattern(dash)) ///
        legend(order(1 "P(casado) observada por edad" ///
                     2 "Ajuste MPL (MCO)" ///
                     3 "Límites válidos de probabilidad [0,1]") ///
               cols(1) size(small) position(5) ring(0)) ///
        note("Nota: La línea roja supera el límite superior (P > 1) para edades altas.", ///
             size(vsmall)) ///
        scheme(s1color)

    graph export "${output}/fig2_mpl_hansen.png", replace width(900)
    di "Gráfico guardado: fig2_mpl_hansen.png"

restore

/*-------------------------------------------------------------------------
 En el do-file utilizamos el ejemplo de Hansen para complementar la clase magistral 
 sobre los modelos lineales de probabilidad, para comentarios y sugerencias 
 comunicarse al correo sadan.de@unipamplona.edu.co
-------------------------------------------------------------------------*/
