import numpy as np
from scipy import signal, optimize
#import sklearn
from statsmodels import regression


class ARMA(object):
    '''Auto-Regressive Moving Average model optimizer
    
        rhoy(L) * y_t = rhoe(L) * eta_t

        Parameters
        ----------
            x : array, 1d
                time series data 
            p : int
                number of AR lags to estimate
            q : int
                number of MA lags to estimate
            rhoy0, rhoe0 : array_like (optional)
                initial parameter values
            optimizer: callable
                optimization function with an interface compatible with scipy.optimize.leastsq

    optimizer = optimize.fmin_bfgs or optimize.leastsq

    TODO:
        1. Consider using an ARIMA (integrated) model to deal with nonstationary systems
           Just need to back-difference x before doing a model fit. 
           Then integrate (sum) the prediction values for comparison to the truth values (x).
        2. Remove all arguments to self.fit, except rhoy0 and rhoe0.
        3. Use a, b lfilter notation rather than ARMA rho and rhoy notation?
        3. Don't prefix paramters with [1] (to allow nonnormalized coeficients).
    '''
    def __init__(self, x=None, p=None, q=None, optimizer=optimize.leastsq):
        self.x = None
        self.p = None
        self.q = 1
        if x is not None:
            self.x = x
        if p is not None:
            self.p = p or self.p
        if q is not None:
            self.q = q or self.q
        self.optimizer = optimizer or optimize.leastsq

    def __repr__(self):
        return 'ARMA(%s, p=%s, q=%s, optimizer=%s, rho=%s, rhoe=%s, rhoy=%s)' % (
            list(self.x[:min(len(self.x),3)]) + (['...'] if len(self.x) > 3 else []), 
            self.p, self.q, self.optimizer, self.rho, self.a, self.b)

    def fit(self, x=None, p=None, q=None, rhoy0=None, rhoe0=None):
        '''Estimate lag coefficients of ARMA process
        
        Returns
        -------
            rh, cov_x, infodict, mesg, ier : output of scipy.optimize.leastsq
            rh :
                estimate of lag parameters, concatenated [rhoy, rhoe]
            cov_x :
                unscaled (!) covariance matrix of coefficient estimates
    
        '''
        if x is not None:
            self.x = x
        if p is not None:
            self.p = p or self.p
        if q is not None:
            self.q = q or self.q
        
        if not rhoy0:
            rhoy0 = 0.5 * np.ones(self.p)
        if not rhoe0:
            rhoe0 = 0.5 * np.ones(self.q)
        rho_combined = np.concatenate((rhoy0, rhoe0))

        if self.optimizer is optimize.leastsq:
            rh, cov_x, infodict, mesg, ier = self.optimizer(
                self.errfn, rho_combined, ftol=1e-10, full_output=True)
        else:
            # fmin_bfgs is slow or doesn't work yet
            errfnsum = lambda rho : np.sum(self.errfn(rho)**2)
            #xopt, {fopt, gopt, Hopt, func_calls, grad_calls
            rh,fopt, gopt, cov_x, _,_, ier = optimize.fmin_bfgs(
                errfnsum, rho_combined, maxiter=2, full_output=True)
            infodict, mesg = None, None
        self.rho = rh
        self.b = np.concatenate(([1], rh[:self.p]))
        self.a = np.concatenate(([1], rh[self.p:])) #rh[-q:])) doesnt work for q=0
        self.error_estimate = self.errfn()
        return rh, cov_x, infodict, mesg, ier
        
    def errfn(self, rho=None, x=None): #, x=None, p=None, q=None):
        ''' duplicate -> remove one
        '''
        if x is not None:
            self.x = x
        if rho is not None:
            self.rho = rho
        # if p is not None:
        #     self.p = p or self.p
        # if q is not None:
        #     self.q = q or self.q
        #[rhoy, rhoe] = [b0, b1... bp, a0, a1 ... aq] = rho
        if self.rho is not None and len(self.rho) >= self.p >= 1:
            #rhoy, rhoe = rho
            self.b = np.concatenate(([1], self.rho[:self.p]))
            self.a = np.concatenate(([1], self.rho[self.p:]))
            return signal.lfilter(self.b, self.a, self.x)
        else: 
            return signal.lfilter(self.b, self.a, self.x)
    
    def forecast(self, ar=None, ma=None, nperiod=10):
        eta = np.r_[self.error_estimate, np.zeros(nperiod)]
        if ar is None:
            ar = self.b
        if ma is None:
            ma = self.a
        return signal.lfilter(ma, ar, eta)      

    def generate_sample(self,ar,ma,nsample,std=1):
        eta = std * np.random.randn(nsample)
        return signal.lfilter(ma, ar, eta)

def generate_sample(self, ar, ma, nsample, std=1, distrvs=np.random.randn):
    eta = std * distrvs(nsample)
    return signal.lfilter(ma, ar, eta)


def impulse_response(ar, ma, nobs=100):
    '''get the impulse response function for ARMA process

    Parameters
    ----------
        ma : array_like
            moving average lag polynomial
        ar : array_like
            auto regressive lag polynomial
        nobs : int
            number of observations to calculate
    
    
    Examples
    --------
    AR(1)
    >>> impulse_response([1.0, -0.8], [1.], nobs=10)
    array([ 1.        ,  0.8       ,  0.64      ,  0.512     ,  0.4096    ,
            0.32768   ,  0.262144  ,  0.2097152 ,  0.16777216,  0.13421773])
    
    this is the same as
    >>> 0.8**np.arange(10)
    array([ 1.        ,  0.8       ,  0.64      ,  0.512     ,  0.4096    ,
            0.32768   ,  0.262144  ,  0.2097152 ,  0.16777216,  0.13421773])

    MA(2)
    >>> impulse_response([1.0], [1., 0.5, 0.2], nobs=10)
    array([ 1. ,  0.5,  0.2,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ])

    ARMA(1,2)
    >>> impulse_response([1.0, -0.8], [1., 0.5, 0.2], nobs=10)
    array([ 1.        ,  1.3       ,  1.24      ,  0.992     ,  0.7936    ,
            0.63488   ,  0.507904  ,  0.4063232 ,  0.32505856,  0.26004685])


    '''
    impulse = np.zeros(nobs)
    impulse[0] = 1.
    return signal.lfilter(ma, ar, impulse)


def mcarma22(niter=10):
    nsample = 1000
    #ar = [1.0, 0, 0]
    ar = [1.0, -0.75, -0.1]
    #ma = [1.0, 0, 0]
    ma = [1.0,  0.3,  0.2]
    results = []
    results_bse = []
    arma = ARMA()
    for _ in range(niter):
        y2 = arma.generate_sample(ar, ma, nsample,0.1)
        rhohat2a, cov_x2a, infodict, mesg, ier = arma.fit(y2,2,2)
        results.append(rhohat2a)
        err2a = arma.errfn(x=y2)
        sige2a = np.sqrt(np.dot(err2a,err2a) / nsample)
        results_bse.append(sige2a * np.sqrt(np.diag(cov_x2a)))
    return np.r_[ar[1:], ma[1:]], np.array(results), np.array(results_bse)
        

if __name__ == '__main__':
    
    # Simulate AR(1)
    #--------------
    # ar * y = ma * eta
    ar = [1., -0.8]
    ma = [1.]
    p, q = 1, 1
    
    # generate AR data
    eta = 0.1 * np.random.randn(1000)
    yar1 = signal.lfilter(ar, ma, eta)
    
    print "\nExample 0"
    arma = ARMA()
    rhohat, cov_x, infodict, mesg, ier = arma.fit(yar1, p, q)
    print 'rho'
    print arma.rho
    print "estimated b, a"
    print arma.b
    print arma.a 
    print 'truth = %s\nestimate = %s\nAR_error = %s\nMA_error = %s' % (ar, rhohat, rhohat[:p] - ar[1:], rhohat[p:] - ma[1:])
    print 'covariance of x'
    print cov_x
    
    print "\nExample 1"
    ar = [1.,  -0.8]
    ma = [1.,  0.5]
    y1 = arma.generate_sample(ar, ma, 1000,0.1)
    arma = ARMA(x=y1, p=1, q=1)
    rhohat1, cov_x1, infodict, mesg, ier = arma.fit(y1, 1, 1)
    print 'estimate'
    print rhohat1
    print 'covariance'
    print cov_x1
    err1 = arma.errfn(x=y1)
    print 'error'
    print np.var(err1)

    print regression.yule_walker(y1, order=2, inv=True)

    print "\nExample 2"
    arma2 = ARMA()
    nsample = 1000
    ar = [1.0, -0.6, -0.1]
    ma = [1.0,  0.3,  0.2]
    y2 = arma2.generate_sample(ar,ma,nsample,0.1)
    rhohat2, cov_x2, infodict, mesg, ier = arma2.fit(y2,1,2)
    print rhohat2
    print cov_x2
    err2 = arma2.errfn(x=y2)
    print 'error = %s' % np.var(err2)
    print "estimated b, a"
    print arma2.b
    print arma2.a
    print "truth"
    print ar
    print ma
    rhohat2a, cov_x2a, infodict, mesg, ier = arma.fit(y2,2,2)
    print rhohat2a
    print cov_x2a
    err2a = arma.errfn(x=y2)
    print 'error = %s' % np.var(err2a)
    print "estimated b, a"
    print arma2.b
    print arma2.a
    print "truth"
    print ar
    print ma
    print 'regression'
    print regression.yule_walker(y2, order=2, inv=True)
    print
    print "\nExample 20"
    arma20 = ARMA()
    nsample = 1000
    ar = [1.0]#, -0.8, -0.4]
    ma = [1.0,  0.5,  0.2]
    y3 = arma20.generate_sample(ar,ma,nsample,0.01)
    rhohat3, cov_x3, infodict, mesg, ier = arma20.fit(y3,2,0)
    print rhohat3
    print cov_x3
    err3 = arma20.errfn(x=y3)
    print np.var(err3)
    print np.sqrt(np.dot(err3,err3)/nsample)
    print "estimated b, a"
    print arma20.b
    print arma20.a
    print "truth"
    print ar
    print ma 
    
    rhohat3a, cov_x3a, infodict, mesg, ier = arma20.fit(y3,0,2)
    print rhohat3a
    print cov_x3a
    err3a = arma20.errfn(x=y3)
    print np.var(err3a)
    print np.sqrt(np.dot(err3a,err3a)/nsample)
    print arma20.b
    print arma20.a
    print "truth"
    print ar
    print ma
   
    print regression.yule_walker(y3, order=2, inv=True)    

    print "\nExample 02"
    arest02 = ARMA()
    nsample = 1000
    ar = [1.0, -0.8, 0.4] #-0.8, -0.4]
    ma = [1.0]#,  0.8,  0.4]
    y4 = arest02.generate_sample(ar,ma,nsample)
    rhohat4, cov_x4, infodict, mesg, ier = arest02.fit(y4, 2, 0)
    print rhohat4
    print cov_x4
    err4 = arest02.errfn(x=y4)
    print np.var(err4)
    sige = np.sqrt(np.dot(err4,err4)/nsample)
    print sige
    print sige * np.sqrt(np.diag(cov_x4))
    print np.sqrt(np.diag(cov_x4))
    print "estimated b, a"
    print arest02.b
    print arest02.a 
    print "truth"
    print ar
    print ma 
    
    rhohat4a, cov_x4a, infodict, mesg, ier = arest02.fit(y4, 0, 2)
    print rhohat4a
    print cov_x4a
    err4a = arest02.errfn(x=y4)
    print np.var(err4a)
    sige = np.sqrt(np.dot(err4a,err4a)/nsample)
    print sige
    print sige * np.sqrt(np.diag(cov_x4a))
    print np.sqrt(np.diag(cov_x4a))
    print "estimated b, a"
    print arest02.b
    print arest02.a  
    print "truth"
    print ar
    print ma
    print regression.yule_walker(y4, order=2, method='mle', inv=True)  

    def mc_summary(res, rt=None):
        if rt is None:
            rt = np.zeros(res.shape[1])
        print 'RMSE'
        print np.sqrt(((res-rt)**2).mean(0))
        print 'mean bias'
        print (res-rt).mean(0)
        print 'median bias'
        print np.median((res-rt),0)
        print 'median bias percent'
        print np.median((res-rt)/rt*100,0)
        print 'median absolute error'
        print np.median(np.abs(res-rt),0)
        print 'positive error fraction'
        print (res > rt).mean(0)

    run_mc = False
    if run_mc:
        import time
        t0 = time.time()
        rt, res_rho, res_bse = mcarma22(niter=1000)
        print 'elapsed time for Monte Carlo', time.time()-t0
        # 20 seconds for ARMA(2,2), 1000 iterations with 1000 observations
        sige2a = np.sqrt(np.dot(err2a,err2a)/nsample)
        print '\nbse of one sample'
        print sige2a * np.sqrt(np.diag(cov_x2a))
        print '\nMC of rho versus true'
        mc_summary(res_rho, rt)
        print '\nMC of bse versus zero'
        mc_summary(res_bse)
        print '\nMC of bse versus std'
        mc_summary(res_bse, res_rho.std(0))
    
    # this seems trivial, because the arma model includes the current sample, so the prediciton will always match the truth
    import matplotlib.pyplot as plt
    N = len(arma2.x)
    forecast = arma2.forecast()
    t = np.arange(N)
    plt.plot(t, np.matrix((forecast[:N], arma2.x[-N:])).transpose())
    plt.xlabel('sample')
    plt.ylabel('x')
    plt.legend(['forecast', 'truth'])
    plt.show()