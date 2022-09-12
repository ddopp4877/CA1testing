/* Created by Language version: 7.7.0 */
/* NOT VECTORIZED */
#define NRN_VECTORIZED 0
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "scoplib_ansi.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__ch_Navp
#define _nrn_initial _nrn_initial__ch_Navp
#define nrn_cur _nrn_cur__ch_Navp
#define _nrn_current _nrn_current__ch_Navp
#define nrn_jacob _nrn_jacob__ch_Navp
#define nrn_state _nrn_state__ch_Navp
#define _net_receive _net_receive__ch_Navp 
#define states states__ch_Navp 
#define trates trates__ch_Navp 
 
#define _threadargscomma_ /**/
#define _threadargsprotocomma_ /**/
#define _threadargs_ /**/
#define _threadargsproto_ /**/
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 static double *_p; static Datum *_ppvar;
 
#define t nrn_threads->_t
#define dt nrn_threads->_dt
#define gmax _p[0]
#define ar2 _p[1]
#define e _p[2]
#define myi _p[3]
#define g _p[4]
#define m _p[5]
#define h _p[6]
#define s _p[7]
#define ena _p[8]
#define ina _p[9]
#define Dm _p[10]
#define Dh _p[11]
#define Ds _p[12]
#define _g _p[13]
#define _ion_ena	*_ppvar[0]._pval
#define _ion_ina	*_ppvar[1]._pval
#define _ion_dinadv	*_ppvar[2]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 /* external NEURON variables */
 extern double celsius;
 /* declaration of user functions */
 static void _hoc_alps(void);
 static void _hoc_alpv(void);
 static void _hoc_bets(void);
 static void _hoc_trap0(void);
 static void _hoc_trates(void);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _p = _prop->param; _ppvar = _prop->dparam;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_ch_Navp", _hoc_setdata,
 "alps_ch_Navp", _hoc_alps,
 "alpv_ch_Navp", _hoc_alpv,
 "bets_ch_Navp", _hoc_bets,
 "trap0_ch_Navp", _hoc_trap0,
 "trates_ch_Navp", _hoc_trates,
 0, 0
};
#define alps alps_ch_Navp
#define alpv alpv_ch_Navp
#define bets bets_ch_Navp
#define trap0 trap0_ch_Navp
 extern double alps( double );
 extern double alpv( double );
 extern double bets( double );
 extern double trap0( double , double , double , double );
 /* declare global and static user variables */
#define Rd Rd_ch_Navp
 double Rd = 0.03;
#define Rg Rg_ch_Navp
 double Rg = 0.01;
#define Rb Rb_ch_Navp
 double Rb = 0.124;
#define Ra Ra_ch_Navp
 double Ra = 0.4;
#define a0s a0s_ch_Navp
 double a0s = 0.0003;
#define gms gms_ch_Navp
 double gms = 0.2;
#define hmin hmin_ch_Navp
 double hmin = 0.5;
#define htau htau_ch_Navp
 double htau = 0;
#define hinf hinf_ch_Navp
 double hinf = 0;
#define mmin mmin_ch_Navp
 double mmin = 0.02;
#define mtau mtau_ch_Navp
 double mtau = 0;
#define minf minf_ch_Navp
 double minf = 0;
#define qq qq_ch_Navp
 double qq = 10;
#define q10 q10_ch_Navp
 double q10 = 2;
#define qg qg_ch_Navp
 double qg = 1.5;
#define qd qd_ch_Navp
 double qd = 1.5;
#define qa qa_ch_Navp
 double qa = 7.2;
#define qinf qinf_ch_Navp
 double qinf = 4;
#define smax smax_ch_Navp
 double smax = 10;
#define sh sh_ch_Navp
 double sh = 15;
#define sinf sinf_ch_Navp
 double sinf = 0;
#define tq tq_ch_Navp
 double tq = -55;
#define thi2 thi2_ch_Navp
 double thi2 = -45;
#define thi1 thi1_ch_Navp
 double thi1 = -45;
#define tha tha_ch_Navp
 double tha = -30;
#define thinf thinf_ch_Navp
 double thinf = -50;
#define taus taus_ch_Navp
 double taus = 0;
#define vvs vvs_ch_Navp
 double vvs = 2;
#define vvh vvh_ch_Navp
 double vvh = -58;
#define vhalfs vhalfs_ch_Navp
 double vhalfs = -60;
#define zetas zetas_ch_Navp
 double zetas = 12;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "sh_ch_Navp", "mV",
 "tha_ch_Navp", "mV",
 "qa_ch_Navp", "mV",
 "Ra_ch_Navp", "/ms",
 "Rb_ch_Navp", "/ms",
 "thi1_ch_Navp", "mV",
 "thi2_ch_Navp", "mV",
 "qd_ch_Navp", "mV",
 "qg_ch_Navp", "mV",
 "Rg_ch_Navp", "/ms",
 "Rd_ch_Navp", "/ms",
 "qq_ch_Navp", "mV",
 "tq_ch_Navp", "mV",
 "thinf_ch_Navp", "mV",
 "qinf_ch_Navp", "mV",
 "vhalfs_ch_Navp", "mV",
 "a0s_ch_Navp", "ms",
 "zetas_ch_Navp", "1",
 "gms_ch_Navp", "1",
 "smax_ch_Navp", "ms",
 "vvh_ch_Navp", "mV",
 "vvs_ch_Navp", "mV",
 "mtau_ch_Navp", "ms",
 "htau_ch_Navp", "ms",
 "taus_ch_Navp", "ms",
 "gmax_ch_Navp", "mho/cm2",
 "ar2_ch_Navp", "1",
 "myi_ch_Navp", "mA/cm2",
 "g_ch_Navp", "mho/cm2",
 0,0
};
 static double delta_t = 0.01;
 static double h0 = 0;
 static double m0 = 0;
 static double s0 = 0;
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "sh_ch_Navp", &sh_ch_Navp,
 "tha_ch_Navp", &tha_ch_Navp,
 "qa_ch_Navp", &qa_ch_Navp,
 "Ra_ch_Navp", &Ra_ch_Navp,
 "Rb_ch_Navp", &Rb_ch_Navp,
 "thi1_ch_Navp", &thi1_ch_Navp,
 "thi2_ch_Navp", &thi2_ch_Navp,
 "qd_ch_Navp", &qd_ch_Navp,
 "qg_ch_Navp", &qg_ch_Navp,
 "mmin_ch_Navp", &mmin_ch_Navp,
 "hmin_ch_Navp", &hmin_ch_Navp,
 "q10_ch_Navp", &q10_ch_Navp,
 "Rg_ch_Navp", &Rg_ch_Navp,
 "Rd_ch_Navp", &Rd_ch_Navp,
 "qq_ch_Navp", &qq_ch_Navp,
 "tq_ch_Navp", &tq_ch_Navp,
 "thinf_ch_Navp", &thinf_ch_Navp,
 "qinf_ch_Navp", &qinf_ch_Navp,
 "vhalfs_ch_Navp", &vhalfs_ch_Navp,
 "a0s_ch_Navp", &a0s_ch_Navp,
 "zetas_ch_Navp", &zetas_ch_Navp,
 "gms_ch_Navp", &gms_ch_Navp,
 "smax_ch_Navp", &smax_ch_Navp,
 "vvh_ch_Navp", &vvh_ch_Navp,
 "vvs_ch_Navp", &vvs_ch_Navp,
 "minf_ch_Navp", &minf_ch_Navp,
 "hinf_ch_Navp", &hinf_ch_Navp,
 "sinf_ch_Navp", &sinf_ch_Navp,
 "mtau_ch_Navp", &mtau_ch_Navp,
 "htau_ch_Navp", &htau_ch_Navp,
 "taus_ch_Navp", &taus_ch_Navp,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(_NrnThread*, _Memb_list*, int);
static void nrn_state(_NrnThread*, _Memb_list*, int);
 static void nrn_cur(_NrnThread*, _Memb_list*, int);
static void  nrn_jacob(_NrnThread*, _Memb_list*, int);
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(_NrnThread*, _Memb_list*, int);
static void _ode_matsol(_NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[3]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"ch_Navp",
 "gmax_ch_Navp",
 "ar2_ch_Navp",
 "e_ch_Navp",
 0,
 "myi_ch_Navp",
 "g_ch_Navp",
 0,
 "m_ch_Navp",
 "h_ch_Navp",
 "s_ch_Navp",
 0,
 0};
 static Symbol* _na_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 14, _prop);
 	/*initialize range parameters*/
 	gmax = 0.01;
 	ar2 = 1;
 	e = 0;
 	_prop->param = _p;
 	_prop->param_size = 14;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_na_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0]._pval = &prop_ion->param[0]; /* ena */
 	_ppvar[1]._pval = &prop_ion->param[3]; /* ina */
 	_ppvar[2]._pval = &prop_ion->param[4]; /* _ion_dinadv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, _NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _ch_Navp_reg() {
	int _vectorized = 0;
  _initlists();
 	ion_reg("na", -10000.);
 	_na_sym = hoc_lookup("na_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 0);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 14, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 ch_Navp C:/Users/ddopp/source/repos/CA1testing/components/mechanisms/ch_Navp.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double _zmexp , _zhexp , _zsexp ;
static int _reset;
static char *modelname = "sodium channel (voltage dependent)";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int trates(double, double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[3], _dlist1[3];
 static int states(_threadargsproto_);
 
double alpv (  double _lv ) {
   double _lalpv;
 _lalpv = 1.0 / ( 1.0 + exp ( ( _lv - vvh - sh ) / vvs ) ) ;
   
return _lalpv;
 }
 
static void _hoc_alpv(void) {
  double _r;
   _r =  alpv (  *getarg(1) );
 hoc_retpushx(_r);
}
 
double alps (  double _lv ) {
   double _lalps;
 _lalps = exp ( 1.e-3 * zetas * ( _lv - vhalfs - sh ) * 9.648e4 / ( 8.315 * ( 273.16 + celsius ) ) ) ;
   
return _lalps;
 }
 
static void _hoc_alps(void) {
  double _r;
   _r =  alps (  *getarg(1) );
 hoc_retpushx(_r);
}
 
double bets (  double _lv ) {
   double _lbets;
 _lbets = exp ( 1.e-3 * zetas * gms * ( _lv - vhalfs - sh ) * 9.648e4 / ( 8.315 * ( 273.16 + celsius ) ) ) ;
   
return _lbets;
 }
 
static void _hoc_bets(void) {
  double _r;
   _r =  bets (  *getarg(1) );
 hoc_retpushx(_r);
}
 
/*CVODE*/
 static int _ode_spec1 () {_reset=0;
 {
   trates ( _threadargscomma_ v , ar2 ) ;
   Dm = ( minf - m ) / mtau ;
   Dh = ( hinf - h ) / htau ;
   Ds = ( sinf - s ) / taus ;
   }
 return _reset;
}
 static int _ode_matsol1 () {
 trates ( _threadargscomma_ v , ar2 ) ;
 Dm = Dm  / (1. - dt*( ( ( ( - 1.0 ) ) ) / mtau )) ;
 Dh = Dh  / (1. - dt*( ( ( ( - 1.0 ) ) ) / htau )) ;
 Ds = Ds  / (1. - dt*( ( ( ( - 1.0 ) ) ) / taus )) ;
  return 0;
}
 /*END CVODE*/
 static int states () {_reset=0;
 {
   trates ( _threadargscomma_ v , ar2 ) ;
    m = m + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / mtau)))*(- ( ( ( minf ) ) / mtau ) / ( ( ( ( - 1.0 ) ) ) / mtau ) - m) ;
    h = h + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / htau)))*(- ( ( ( hinf ) ) / htau ) / ( ( ( ( - 1.0 ) ) ) / htau ) - h) ;
    s = s + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / taus)))*(- ( ( ( sinf ) ) / taus ) / ( ( ( ( - 1.0 ) ) ) / taus ) - s) ;
   }
  return 0;
}
 
static int  trates (  double _lvm , double _la2 ) {
   double _la , _lb , _lc , _lqt ;
 _lqt = pow( q10 , ( ( celsius - 24.0 ) / 10.0 ) ) ;
   _la = trap0 ( _threadargscomma_ _lvm , tha + sh , Ra , qa ) ;
   _lb = trap0 ( _threadargscomma_ - _lvm , - tha - sh , Rb , qa ) ;
   mtau = 1.0 / ( _la + _lb ) / _lqt ;
   if ( mtau < mmin ) {
     mtau = mmin ;
     }
   minf = _la / ( _la + _lb ) ;
   _la = trap0 ( _threadargscomma_ _lvm , thi1 + sh , Rd , qd ) ;
   _lb = trap0 ( _threadargscomma_ - _lvm , - thi2 - sh , Rg , qg ) ;
   htau = 1.0 / ( _la + _lb ) / _lqt ;
   if ( htau < hmin ) {
     htau = hmin ;
     }
   hinf = 1.0 / ( 1.0 + exp ( ( _lvm - thinf - sh ) / qinf ) ) ;
   _lc = alpv ( _threadargscomma_ _lvm ) ;
   sinf = _lc + _la2 * ( 1.0 - _lc ) ;
   taus = bets ( _threadargscomma_ _lvm ) / ( a0s * ( 1.0 + alps ( _threadargscomma_ _lvm ) ) ) ;
   if ( taus < smax ) {
     taus = smax ;
     }
    return 0; }
 
static void _hoc_trates(void) {
  double _r;
   _r = 1.;
 trates (  *getarg(1) , *getarg(2) );
 hoc_retpushx(_r);
}
 
double trap0 (  double _lv , double _lth , double _la , double _lq ) {
   double _ltrap0;
 if ( fabs ( _lv - _lth ) > 1e-6 ) {
     _ltrap0 = _la * ( _lv - _lth ) / ( 1.0 - exp ( - ( _lv - _lth ) / _lq ) ) ;
     }
   else {
     _ltrap0 = _la * _lq ;
     }
   
return _ltrap0;
 }
 
static void _hoc_trap0(void) {
  double _r;
   _r =  trap0 (  *getarg(1) , *getarg(2) , *getarg(3) , *getarg(4) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 3;}
 
static void _ode_spec(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ena = _ion_ena;
     _ode_spec1 ();
  }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 3; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 ();
 }
 
static void _ode_matsol(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ena = _ion_ena;
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_na_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_na_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_na_sym, _ppvar, 2, 4);
 }

static void initmodel() {
  int _i; double _save;_ninits++;
 _save = t;
 t = 0.0;
{
  h = h0;
  m = m0;
  s = s0;
 {
   trates ( _threadargscomma_ v , ar2 ) ;
   m = minf ;
   h = hinf ;
   s = sinf ;
   }
  _sav_indep = t; t = _save;

}
}

static void nrn_init(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
  ena = _ion_ena;
 initmodel();
 }}

static double _nrn_current(double _v){double _current=0.;v=_v;{ {
   g = gmax * m * m * m * h * s ;
   ina = g * ( v - ena ) ;
   myi = ina ;
   }
 _current += ina;

} return _current;
}

static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
  ena = _ion_ena;
 _g = _nrn_current(_v + .001);
 	{ double _dina;
  _dina = ina;
 _rhs = _nrn_current(_v);
  _ion_dinadv += (_dina - ina)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ina += ina ;
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}}

static void nrn_jacob(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}}

static void nrn_state(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
  ena = _ion_ena;
 { error =  states();
 if(error){fprintf(stderr,"at line 86 in file ch_Navp.mod:\n	SOLVE states METHOD cnexp\n"); nrn_complain(_p); abort_run(error);}
 } }}

}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = &(m) - _p;  _dlist1[0] = &(Dm) - _p;
 _slist1[1] = &(h) - _p;  _dlist1[1] = &(Dh) - _p;
 _slist1[2] = &(s) - _p;  _dlist1[2] = &(Ds) - _p;
_first = 0;
}

#if NMODL_TEXT
static const char* nmodl_filename = "ch_Navp.mod";
static const char* nmodl_file_text = 
  "TITLE sodium channel (voltage dependent)\n"
  "\n"
  "COMMENT\n"
  "sodium channel (voltage dependent)\n"
  "\n"
  "Ions: na\n"
  "\n"
  "Style: quasi-ohmic\n"
  "\n"
  "From: modified from Jeff Magee. M.Migliore may97\n"
  "\n"
  "Updates:\n"
  "2002 April (Michele Migliore): added sh to account for higher threshold\n"
  "2014 December (Marianne Bezaire): documented\n"
  "ENDCOMMENT\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX ch_Navp\n"
  "	USEION na READ ena WRITE ina\n"
  "	RANGE  gmax, ar2, myi, e, g\n"
  "	GLOBAL minf, hinf, mtau, htau, sinf, taus,qinf, thinf\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	sh   = 15		(mV)\n"
  "	gmax = 0.010	(mho/cm2)	\n"
  "								\n"
  "	tha  = -30 		(mV)\n"
  "	qa   = 7.2		(mV)	: act slope		\n"
  "	Ra   = 0.4		(/ms)	: open (v)		\n"
  "	Rb   = 0.124 	(/ms)	: close (v)		\n"
  "\n"
  "	thi1  = -45		(mV)	: v 1/2 for inact 	\n"
  "	thi2  = -45 	(mV)	: v 1/2 for inact 	\n"
  "	qd   = 1.5		(mV)    : inact tau slope\n"
  "	qg   = 1.5      (mV)\n"
  "	mmin = 0.02	\n"
  "	hmin = 0.5			\n"
  "	q10  = 2\n"
  "	Rg   = 0.01 	(/ms)	: inact recov (v) 	\n"
  "	Rd   = 0.03 	(/ms)	: inact (v)	\n"
  "	qq   = 10		(mV)\n"
  "	tq   = -55      (mV)\n"
  "\n"
  "	thinf  = -50 	(mV)	: inact inf slope	\n"
  "	qinf  = 4 		(mV)	: inact inf slope \n"
  "\n"
  "	vhalfs = -60	(mV)	: slow inact.\n"
  "	a0s = 0.0003	(ms)	: a0s=b0s\n"
  "	zetas = 12		(1)\n"
  "	gms = 0.2		(1)\n"
  "	smax = 10		(ms)\n"
  "	vvh = -58		(mV) \n"
  "	vvs = 2			(mV)\n"
  "	ar2 = 1			(1)		: 1=no inact., 0=max inact.\n"
  "	ena				(mV)    : must be explicitly def. in hoc\n"
  "	celsius\n"
  "	v 				(mV)\n"
  "	e\n"
  "}\n"
  "\n"
  "\n"
  "UNITS {\n"
  "	(mA) = (milliamp)\n"
  "	(mV) = (millivolt)\n"
  "	(pS) = (picosiemens)\n"
  "	(um) = (micron)\n"
  "} \n"
  "\n"
  "ASSIGNED {\n"
  "	ina 		(mA/cm2)\n"
  "	myi 		(mA/cm2)\n"
  "	g			(mho/cm2)\n"
  "	minf\n"
  "	hinf 		\n"
  "	sinf\n"
  "	mtau		(ms)\n"
  "	htau		(ms) 	\n"
  "	taus		(ms)\n"
  "}\n"
  " \n"
  "\n"
  "STATE { m h s}\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE states METHOD cnexp\n"
  "	g = gmax*m*m*m*h*s\n"
  "	ina = g * (v - ena)\n"
  "	myi = ina\n"
  "} \n"
  "\n"
  "INITIAL {\n"
  "	trates(v,ar2)\n"
  "	m=minf  \n"
  "	h=hinf\n"
  "	s=sinf\n"
  "}\n"
  "\n"
  "\n"
  "FUNCTION alpv(v(mV)) {\n"
  "	alpv = 1/(1+exp((v-vvh-sh)/vvs))\n"
  "}\n"
  "        \n"
  "FUNCTION alps(v(mV)) {  \n"
  "	alps = exp(1.e-3*zetas*(v-vhalfs-sh)*9.648e4/(8.315*(273.16+celsius)))\n"
  "}\n"
  "\n"
  "FUNCTION bets(v(mV)) {\n"
  "	bets = exp(1.e-3*zetas*gms*(v-vhalfs-sh)*9.648e4/(8.315*(273.16+celsius)))\n"
  "}\n"
  "\n"
  "LOCAL mexp, hexp, sexp\n"
  "\n"
  "DERIVATIVE states {   \n"
  "	trates(v,ar2)      \n"
  "	m' = (minf-m)/mtau\n"
  "	h' = (hinf-h)/htau\n"
  "	s' = (sinf - s)/taus\n"
  "}\n"
  "\n"
  "PROCEDURE trates(vm,a2) {  \n"
  "	LOCAL  a, b, c, qt\n"
  "	qt=q10^((celsius-24)/10)\n"
  "	a = trap0(vm,tha+sh,Ra,qa)\n"
  "	b = trap0(-vm,-tha-sh,Rb,qa)\n"
  "	mtau = 1/(a+b)/qt\n"
  "	if (mtau<mmin) {mtau=mmin}\n"
  "	minf = a/(a+b)\n"
  "\n"
  "	a = trap0(vm,thi1+sh,Rd,qd)\n"
  "	b = trap0(-vm,-thi2-sh,Rg,qg)\n"
  "	htau =  1/(a+b)/qt\n"
  "	if (htau<hmin) {htau=hmin}\n"
  "	hinf = 1/(1+exp((vm-thinf-sh)/qinf))\n"
  "	c=alpv(vm)\n"
  "	sinf = c+a2*(1-c)\n"
  "	taus = bets(vm)/(a0s*(1+alps(vm)))\n"
  "	if (taus<smax) {taus=smax}\n"
  "}\n"
  "\n"
  "FUNCTION trap0(v,th,a,q) {\n"
  "	if (fabs(v-th) > 1e-6) {\n"
  "	    trap0 = a * (v - th) / (1 - exp(-(v - th)/q))\n"
  "	} else {\n"
  "	    trap0 = a * q\n"
  " 	}\n"
  "}	\n"
  ;
#endif