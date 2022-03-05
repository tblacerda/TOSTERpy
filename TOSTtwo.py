import pandas as pd
import numpy as np
from numpy import sqrt as sqrt
import scipy.stats as stats
from scipy.stats import t as student_t
from statistics import stdev



def TOSTtwo(m1, m2,
            sd1, sd2,
            n1, n2,
            low_eqbound_d,
            high_eqbound_d,
            alpha = 0.05,
            var_equal = False,
            plot = False,
            verbose = True):
    '''
    TOST function for an independent t-test (Cohen's d)
    @param m1 mean of group 1
    @param m2 mean of group 2
    @param sd1 standard deviation of group 1
    @param sd2 standard deviation of group 2
    @param n1 sample size in group 1
    @param n2 sample size in group 2
    @param low_eqbound_d lower equivalence bounds (e.g., -0.5) expressed in standardized mean 
    difference (Cohen's d)
    @param high_eqbound_d upper equivalence bounds (e.g., 0.5) expressed in standardized mean 
    difference (Cohen's d)
    @param alpha alpha level (default = 0.05)
    @param var.equal logical variable indicating whether equal variances assumption is assumed to 
    be TRUE or FALSE.  Defaults to FALSE.
    @param plot set whether results should be plotted (plot = TRUE) or not (plot = FALSE) - defaults
     to TRUE
    @param verbose logical variable indicating whether text output should be generated (verbose 
    = TRUE) or not (verbose = FALSE) - default to TRUE
    @return Returns TOST t-value 1, TOST p-value 1, TOST t-value 2, TOST p-value 2, degrees of 
    freedom, low equivalence bound, high equivalence bound, low equivalence bound in Cohen's d,
     high equivalence bound in Cohen's d, Lower limit confidence interval TOST, Upper limit 
     confidence interval TOST
    @importFrom stats pnorm pt qnorm qt
    @importFrom graphics abline plot points segments title
    @examples
    ## Eskine (2013) showed that participants who had been exposed to organic
    ## food were substantially harsher in their moral judgments relative to
    ## those exposed to control (d = 0.81, 95% CI: [0.19, 1.45]). A
    ## replication by Moery & Calin-Jageman (2016, Study 2) did not observe
    ## a significant effect (Control: n = 95, M = 5.25, SD = 0.95, Organic
    ## Food: n = 89, M = 5.22, SD = 0.83). Following Simonsohn's (2015)
    ## recommendation the equivalence bound was set to the effect size the
    ## original study had 33% power to detect (with n = 21 in each condition,
    ## this means the equivalence bound is d = 0.48, which equals a
    ## difference of 0.384 on a 7-point scale given the sample sizes and a
    ## pooled standard deviation of 0.894). Using a TOST equivalence test
    ## with default alpha = 0.05, not assuming equal variances, and equivalence
    ## bounds of d = -0.43 and d = 0.43 is significant, t(182) = -2.69,
    ## p = 0.004. We can reject effects larger than d = 0.43.
    TOSTtwo(m1=5.25,m2=5.22,sd1=0.95,sd2=0.83,n1=95,n2=89,low_eqbound_d=-0.43,high_eqbound_d=0.43)
    @section References:
    Berger, R. L., & Hsu, J. C. (1996). Bioequivalence Trials, Intersection-Union Tests
     and Equivalence Confidence Sets. Statistical Science, 11(4), 283-302.
    Gruman, J. A., Cribbie, R. A., & Arpin-Cribbie, C. A. (2007).
     The effects of heteroscedasticity on tests of equivalence. 
     Journal of Modern Applied Statistical Methods, 6(1), 133-140, 
     formula for Welch's t-test on page 135
    
    THIS FUNCTION WAS REPLICATED FOR EDUCATIONAL PURPOSES AS IT WAS REPLACED BY
    tsum_TOST wich is better designed and has a broader usage.

    Return list

    '''

    if  (n1 < 2) or (n2 < 2):
        return "The sample size should be larger than 1."

    if (1<=alpha or alpha < 0):
        return "The alpha level should be a positive value between 0 and 1."
    
    if (sd1 <= 0 or sd2 <=0):
        return "The standard deviation should be a positive value."
    
    ## Fim dos checks
      # Calculate TOST, t-test, 90% CIs and 95% CIs
      
    if var_equal == True:
        sdpooled = sqrt((((n1 - 1)*(sd1**2))+(n2 - 1)*(sd2**2))/((n1+n2)-2))
        low_eqbound = low_eqbound_d*sdpooled
        high_eqbound = high_eqbound_d*sdpooled
        degree_f = n1+n2-2
    
        dist = student_t(df=degree_f,loc=0,scale=1 )


        t1 = ((m1-m2)-low_eqbound)/(sdpooled*sqrt(1/n1 + 1/n2))  #students t-test lower bound
        lower_tail_false = 1- dist.cdf(t1)  
        p1 = lower_tail_false 
        t2 = ((m1-m2)-high_eqbound)/(sdpooled*sqrt(1/n1 + 1/n2)) #students t-test upper bound
        lower_tail_true = dist.cdf(t2)
        p2 = lower_tail_true
        
        t = (m1-m2)/(sdpooled*sqrt(1/n1 + 1/n2))
        
        lower_tail_true2 = dist.cdf(-abs(t))
        pttest = 2*lower_tail_true2
        
        LL90 = (m1-m2)-student_t.ppf(1-alpha, n1+n2-2)*(sdpooled*sqrt(1/n1 + 1/n2))
        UL90 = (m1-m2)+student_t.ppf(1-alpha, n1+n2-2)*(sdpooled*sqrt(1/n1 + 1/n2))
        LL95 = (m1-m2)-student_t.ppf(1-(alpha/2), n1+n2-2)*(sdpooled*sqrt(1/n1 + 1/n2))
        UL95 = (m1-m2)+student_t.ppf(1-(alpha/2), n1+n2-2)*(sdpooled*sqrt(1/n1 + 1/n2))
    else:
        sdpooled = sqrt((sd1**2 + sd2**2)/2) #calculate sd root mean squared for Welch's t-test
        low_eqbound = low_eqbound_d*sdpooled
        high_eqbound = high_eqbound_d*sdpooled
        degree_f = (sd1**2/n1+sd2**2/n2)**2/(((sd1**2/n1)**2/(n1-1))+((sd2**2/n2)**2/(n2-1))) #degrees of freedom for Welch's t-test        
        dist = student_t(df=degree_f,loc=0,scale=1 )
        t1 = ((m1-m2)-low_eqbound)/sqrt(sd1**2/n1 + sd2**2/n2) #welch's t-test upper bound
        lower_tail_false = 1- dist.cdf(t1)  
        p1 = lower_tail_false 
        t2 = ((m1-m2)-high_eqbound)/sqrt(sd1**2/n1 + sd2**2/n2) #welch's t-test lower bound
        lower_tail_true = dist.cdf(t2)
        p2 = lower_tail_true
        t = (m1-m2)/sqrt(sd1**2/n1 + sd2**2/n2) #welch's t-test NHST    
        lower_tail_true2 = dist.cdf(-abs(t))
        pttest = 2*lower_tail_true2
    
        LL90 = (m1-m2)-student_t.ppf(1-alpha, degree_f)*sqrt(sd1**2/n1 + sd2**2/n2) #Lower limit for CI Welch's t-test
        UL90 = (m1-m2)+student_t.ppf(1-alpha, degree_f)*sqrt(sd1**2/n1 + sd2**2/n2) #Upper limit for CI Welch's t-test
        LL95 = (m1-m2)-student_t.ppf(1-(alpha/2), degree_f)*sqrt(sd1**2/n1 + sd2**2/n2) #Lower limit for CI Welch's t-test
        UL95 = (m1-m2)+student_t.ppf(1-(alpha/2), degree_f)*sqrt(sd1**2/n1 + sd2**2/n2) #Upper limit for CI Welch's t-test
  
    ptost = max(p1,p2) #Get highest p-value for summary TOST result
    ttost = t2
    if (abs(t1) < abs(t2)):
        ttost = t1
  
    dif = (m1-m2)
    testoutcome = "non-significant"
    
    if pttest < alpha:
        testoutcome = "significant"
    
    TOSToutcome = "non-significant"
    if ptost<alpha:
        TOSToutcome = "significant"
    
    if verbose == True:

        print("TOST Results:")
        print(80*"=")
        print("t-value lower bound: %0.4f ; tp-value lower bound: %0.4f"%(t1, p1))
        print("t-value upper bound: %0.4f ; tp-value upper bound: %0.4f"%(t2, p2))
        print("Degrees of freedom: %0.2f"%(round(degree_f, 2)))
        print("Equivalence bounds (Cohen's d): low eqbound: %0.4f ; high eqbound: %0.4f"%(low_eqbound_d, high_eqbound_d))
        print("TOST confidence interval: lower bound %0.4f CI: %0.4f; upper bound %0.4f CI: %0.4f"%((100*1-alpha*2),round(LL90,3),(100*1-alpha*2),round(UL90,3)))
        print("NHST confidence interval: lower bound %0.4f CI: %0.4f; upper bound %0.4f CI: %0.4f"%((100*1-alpha),round(LL95,3),(100*1-alpha),round(UL95,3)))
        print("\nEquivalence Test Result:")
        print(80*"=")
        print("The equivalence test was %s, t(%0.2f) = %0.4f, p = %0.4f, given equivalence bounds of %0.4f and %0.4f (on a raw scale) and an alpha of %0.3f"%(TOSToutcome, degree_f,ttost, ptost, low_eqbound, high_eqbound, alpha))
        
        print("\nNull Hypothesis Test Result:")
        print(80*"=")
        print("The null hypothesis test was %s, t(%0.4f) = %0.4f, p = %0.4f, given an alpha of %0.3f"%(testoutcome, degree_f, t, pttest, alpha))

        if (pttest <= alpha and ptost <= alpha):
            combined_outcome = "NHST: reject null significance hypothesis that the effect is equal to 0. \n TOST: reject null equivalence hypothesis."
        
        if (pttest < alpha and ptost > alpha):
            combined_outcome = "NHST: reject null significance hypothesis that the effect is equal to 0. \n TOST: Don't reject null equivalence hypothesis."

        if (pttest > alpha and ptost <= alpha):
            combined_outcome = "NHST: Don't reject null significance hypothesis that the effect is equal to 0. \n TOST: reject null equivalence hypothesis."
        
        if (pttest > alpha and ptost > alpha):
            combined_outcome = "NHST: Don't reject null significance hypothesis that the effect is equal to 0. \n TOST: Don't reject null equivalence hypothesis."
        print("\nOutcome:\n %s"%(combined_outcome))
        print(80*"=")

    return [dif, t1, p1, t2, p2, degree_f, low_eqbound, high_eqbound, low_eqbound_d, high_eqbound_d,
            LL90, UL90, LL95, UL95, t, pttest]

 

# TOSTtwo(m1 = 4.55,
#         m2 = 4.87,
#         sd1 = 1.05,
#         sd2 = 1.11,
#         n1 = 150,
#         n2 = 15,
#         low_eqbound_d= -0.5,
#         high_eqbound_d= 0.5)



TOSTtwo(m1 = 6.492733119, m2 = 6.448293963, sd1 = 0.88388498, sd2 = 1.431852215, n1 = 80, n2 =80, low_eqbound_d = -0.4, high_eqbound_d = .4)



