Search.setIndex({docnames:["index","privugger","privugger.attacker","privugger.datastructures","privugger.distributions","privugger.inference","privugger.measures","privugger.transformer","tutorials/Governor","tutorials/Open-dp-Tutorial","tutorials/Tutorial"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.todo":2,"sphinx.ext.viewcode":1,nbsphinx:3,sphinx:56},filenames:["index.rst","privugger.rst","privugger.attacker.rst","privugger.datastructures.rst","privugger.distributions.rst","privugger.inference.rst","privugger.measures.rst","privugger.transformer.rst","tutorials/Governor.ipynb","tutorials/Open-dp-Tutorial.ipynb","tutorials/Tutorial.ipynb"],objects:{"":{privugger:[1,0,0,"-"]},"privugger.attacker":{distributions:[2,0,0,"-"],generators:[2,0,0,"-"],metrics:[2,0,0,"-"]},"privugger.attacker.distributions":{MINIMUM_COVERAGE:[2,1,1,""],Support:[2,2,1,""]},"privugger.attacker.distributions.Support":{BERNOULLI:[2,3,1,""],BETA:[2,3,1,""],BETA_BINOMIAL:[2,3,1,""],BINOMIAL:[2,3,1,""],CAUCHY:[2,3,1,""],DISCRETE_UNIFORM:[2,3,1,""],EXPONENTIAL:[2,3,1,""],GAMMA:[2,3,1,""],GEOMETRIC:[2,3,1,""],LAPLACE:[2,3,1,""],NORMAL:[2,3,1,""],POISSON:[2,3,1,""],STUDENT_T:[2,3,1,""],TRUNCATED_NORMAL:[2,3,1,""],UNIFORM:[2,3,1,""]},"privugger.attacker.generators":{Bernoulli:[2,1,1,""],Beta:[2,1,1,""],BetaBinomial:[2,1,1,""],Binomial:[2,1,1,""],Cauchy:[2,1,1,""],DiscreteUniform:[2,1,1,""],Exponential:[2,1,1,""],FloatGenerator:[2,1,1,""],FloatList:[2,1,1,""],Gamma:[2,1,1,""],Geometric:[2,1,1,""],IntGenerator:[2,1,1,""],IntList:[2,1,1,""],Laplace:[2,1,1,""],Normal:[2,1,1,""],Poisson:[2,1,1,""],StudentT:[2,1,1,""],TruncatedNormal:[2,1,1,""],Uniform:[2,1,1,""]},"privugger.attacker.metrics":{SimulationMetrics:[2,2,1,""]},"privugger.attacker.metrics.SimulationMetrics":{highest_leakage:[2,4,1,""],load_from_file:[2,4,1,""],mutual_information:[2,4,1,""],plot_mutual_bar:[2,4,1,""],plot_mutual_information:[2,4,1,""],save_to_file:[2,4,1,""]},"privugger.distributions":{continuous:[4,0,0,"-"],discrete:[4,0,0,"-"]},"privugger.distributions.continuous":{Beta:[4,2,1,""],Exponential:[4,2,1,""],Normal:[4,2,1,""],Uniform:[4,2,1,""]},"privugger.distributions.continuous.Beta":{alpha:[4,3,1,""],beta:[4,3,1,""],get_params:[4,4,1,""],is_hyper_param:[4,3,1,""],name:[4,3,1,""],num_elements:[4,3,1,""],pymc3_dist:[4,4,1,""],scipy_dist:[4,4,1,""]},"privugger.distributions.continuous.Exponential":{get_params:[4,4,1,""],is_hyper_param:[4,3,1,""],lam:[4,3,1,""],name:[4,3,1,""],num_elements:[4,3,1,""],pymc3_dist:[4,4,1,""],scipy_dist:[4,4,1,""]},"privugger.distributions.continuous.Normal":{get_params:[4,4,1,""],is_hyper_param:[4,3,1,""],mu:[4,3,1,""],name:[4,3,1,""],num_elements:[4,3,1,""],pymc3_dist:[4,4,1,""],scipy_dist:[4,4,1,""],std:[4,3,1,""]},"privugger.distributions.continuous.Uniform":{get_params:[4,4,1,""],is_hyper_param:[4,3,1,""],lower:[4,3,1,""],name:[4,3,1,""],num_elements:[4,3,1,""],pymc3_dist:[4,4,1,""],scipy_dist:[4,4,1,""],upper:[4,3,1,""]},"privugger.distributions.discrete":{Bernoulli:[4,2,1,""],Binomial:[4,2,1,""],Categorical:[4,2,1,""],Constant:[4,2,1,""],DiscreteUniform:[4,2,1,""],Geometric:[4,2,1,""]},"privugger.distributions.discrete.Bernoulli":{get_params:[4,4,1,""],is_hyper_param:[4,3,1,""],name:[4,3,1,""],num_elements:[4,3,1,""],p:[4,3,1,""],pymc3_dist:[4,4,1,""],scipy_dist:[4,4,1,""]},"privugger.distributions.discrete.Binomial":{get_params:[4,4,1,""],is_hyper_param:[4,3,1,""],n:[4,3,1,""],name:[4,3,1,""],num_elements:[4,3,1,""],p:[4,3,1,""],pymc3_dist:[4,4,1,""],scipy_dist:[4,4,1,""]},"privugger.distributions.discrete.Categorical":{get_params:[4,4,1,""],is_hyper_param:[4,3,1,""],name:[4,3,1,""],num_elements:[4,3,1,""],p:[4,3,1,""],pymc3_dist:[4,4,1,""],scipy_dist:[4,4,1,""]},"privugger.distributions.discrete.Constant":{get_params:[4,4,1,""],is_hyper_param:[4,3,1,""],name:[4,3,1,""],num_elements:[4,3,1,""],pymc3_dist:[4,4,1,""],scipy_dist:[4,4,1,""],val:[4,3,1,""]},"privugger.distributions.discrete.DiscreteUniform":{get_params:[4,4,1,""],is_hyper_param:[4,3,1,""],lower:[4,3,1,""],name:[4,3,1,""],num_elements:[4,3,1,""],pymc3_dist:[4,4,1,""],scipy_dist:[4,4,1,""],upper:[4,3,1,""]},"privugger.distributions.discrete.Geometric":{get_params:[4,4,1,""],is_hyper_param:[4,3,1,""],name:[4,3,1,""],num_elements:[4,3,1,""],p:[4,3,1,""],pymc3_dist:[4,4,1,""],scipy_dist:[4,4,1,""]},"privugger.measures":{mutual_information:[6,0,0,"-"]},"privugger.measures.mutual_information":{mi_sklearn:[6,1,1,""]},privugger:{data_structures:[3,0,0,"-"],inference:[5,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","attribute","Python attribute"],"4":["py","method","Python method"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:attribute","4":"py:method"},terms:{"0":[2,4,6,8,9,10],"00":[8,9,10],"0002":10,"00190783":10,"005":9,"0087944217930924":10,"01108224920021":10,"01775":8,"02":9,"03":8,"035255029027044":10,"05":10,"06":10,"1":[2,4,6,8,9,10],"10":[2,9,10],"100":[2,8,9,10],"10000":8,"100k":9,"10_000":[8,10],"11":[2,10],"12":[2,10],"120":9,"13":10,"14":10,"15":8,"150":9,"16":[2,8],"180":9,"1_000":[8,9,10],"2":[2,4,8,9,10],"20":[6,8,9],"200":[8,9],"20000":9,"2002":8,"202":8,"20235":8,"20_000":[8,9,10],"21":8,"22000":[8,10],"227":[8,9],"25":8,"271":10,"2_000":[8,9,10],"3":[2,8,9,10],"31":10,"35":10,"37":10,"38":10,"39":[8,10],"392":10,"4":[2,8,9,10],"40":[9,10],"40_000":9,"42":10,"42000":9,"44":10,"45":10,"46":10,"49245":10,"5":[4,8,9,10],"50":[8,9],"5001":10,"50755":10,"55":[8,10],"561":[8,9],"59":9,"6":[2,8,9,10],"60":[8,9],"7":[2,8,9,10],"8":[2,8,9,10],"80921166":10,"9":[2,10],"95":10,"95512445":10,"95685":8,"98105":8,"\u00b5":2,"\u00df":2,"boolean":[4,6],"case":[0,2,9,10],"class":[2,4],"default":[4,6],"do":10,"final":[8,9,10],"float":[2,4,6,8,9,10],"function":[6,8,9,10],"import":[8,9,10],"int":[2,4,6,9],"new":10,"pr\u026av\u028c\u0261\u0259":0,"return":[0,2,6,8,9,10],"true":[2,6,9,10],"var":10,A:[2,8,10],As:[8,10],At:[8,10],For:[9,10],If:[0,6,10],In:[0,2,8,9,10],Is:2,That:[0,8,10],The:[1,2,4,8,9,10],There:10,These:[8,10],To:9,_hdi:10,about:[0,2,9],abov:[8,9,10],access:[9,10],accord:9,accur:10,accuraci:9,acknowledg:8,action:10,adapt:9,adapt_diag:10,add:[8,9,10],add_observ:10,addition:10,adversari:[0,8],after:[8,9,10],ag:[8,9,10],age_0:10,age_mean:9,ages_db:8,ages_dim_0:10,ages_oth:8,alic:10,alice_ag:8,alice_diagnosi:8,alice_gend:8,alice_nam:8,alice_xxx:8,alice_zip:8,all:[2,8,9,10],allow:[9,10],alpha:[2,4],alreadi:10,also:[2,8,10],alwai:9,among:9,amount:[0,10],an:[8,9,10],analys:[0,2],analysi:0,analyst:2,analyz:[8,9],ani:[9,10],anyth:10,api:[8,9,10],append:[2,8,9,10],applic:8,ar:[2,8,9,10],arbitrari:9,arbitrarili:9,around:8,arrai:[6,8,9,10],arviz:[8,9,10],as_bar:2,assert:9,assign:10,assum:[9,10],assumpt:[8,10],attack:[0,1,9],attr:[8,9],attr_0:9,attribut:9,auto:10,automat:[0,9,10],avail:0,averag:[8,10],avg:[8,10],axi:[8,9],az:[6,8,9,10],b:[2,10],back:10,backend:[8,9,10],bar:[2,8],base:[0,2,4,9,10],bay:0,becaus:9,been:8,being:[8,10],believ:10,below:[8,9,10],bernoulli:[2,4,8],best:10,beta:[2,4],beta_binomi:2,betabinomi:2,better:10,between:[9,10],between_chain_vari:[8,9],binari:10,binarygibbsmetropoli:8,bind:6,binomi:[2,4,9],bla:[8,9,10],bool:[2,6],both:8,bottom:8,bound:4,box:9,boxplot:9,built:8,c:[8,9,10],calcul:2,call:[8,9,10],calul:9,can:[2,8,9,10],categor:4,categori:[9,10],cauchi:2,cell:10,certain:10,chain:[8,9,10],chanc:8,chang:10,choos:[2,10],chose:10,chosen:[2,9],cleaner:10,close:[8,9],code:[8,9,10],color:10,column:[8,9],column_nam:9,combin:8,come:10,comment:9,compar:10,compat:10,complet:[8,10],compos:9,composit:8,composition:8,compoundstep:[8,9],comput:[6,8,9,10],concaten:[8,9],concentr:9,concept:10,conclus:10,concret:[6,10],conform:9,consequ:10,consid:[8,9,10],consist:10,constant:[4,8,9],construct:2,constructor:[9,10],contain:[2,6,8,9,10],containint:2,continu:[1,8,10],converg:[8,9],convert:2,coord:10,core:[8,9,10],correspond:[9,10],count_gov_gend:8,count_gov_genders_zip:8,count_gov_zip:8,count_nonzero:8,creat:[8,10],csv:9,current:[8,10],d:2,data:[2,8,9],data_low:9,data_row:9,data_structur:1,data_upp:9,datafram:9,dataset:[8,10],datastructur:1,de:6,def:[8,9,10],defin:[8,9,10],definit:9,denot:9,describ:2,desir:2,determin:2,detmin:2,deviat:[4,9,10],df:9,diagnos:8,diagnoses_db:8,diagnoses_oth:8,diagnosi:8,diagnost:[8,9],did:[8,9,10],differ:[2,8,10],differenti:9,dimension:2,directli:8,directori:[9,10],disabl:10,disc_featur:6,discard:10,disceret:6,discret:[1,8],discrete_uniform:2,discreteuniform:[2,4,8,9],dist:2,distanc:10,distant:10,distribut:[1,8,9],ditribut:2,diverg:[0,8,9,10],dkk:9,doe:[0,8],doesn:8,domain:[6,10],done:9,dot:2,double_scalar:[8,9],dp_mean:9,dp_program:9,draw:[2,8,9,10],ds:[8,9,10],due:10,e:[8,9,10],each:[2,8,9,10],earlier:10,educ:9,effect:[0,8,9,10],element:[6,10],elementwis:10,els:10,encapsul:10,encount:[8,9],end:9,ensur:8,entropi:0,epsilon:9,equal:[8,10],equip:0,estim:[6,8,9,10],everyth:10,exact:9,exactli:6,exampl:[8,10],exemplifi:8,exhibit:0,exist:[8,10],expect:8,explain:10,explor:8,exponenti:[2,4],extern:10,extract:10,fact:8,fail:10,fall:10,fals:[4,6,8,9,10],famou:8,femal:8,figsiz:2,figur:9,file:[2,10],find:0,first:[8,9,10],fix:8,flatten:10,float64:9,floatgener:2,floatlist:2,focu:8,follow:[0,8],form:[8,10],format:2,from:[2,6,8,10],furthermor:[0,9],futurewarn:10,g:10,gamma:2,gaussian:4,gender:8,genders_db:8,genders_oth:8,gener:[1,10],geometr:[2,4],get:[0,9],get_param:4,getattr:[8,9],give:[4,8,9,10],given:0,global:2,gov:8,governor:0,graph:2,greater:10,group:2,gt:[8,9,10],h:2,had:10,have:[2,8,9,10],hdi:10,hdi_prob:10,head:2,healthi:8,her:8,here:[8,9,10],high:[2,9],highest:[2,9],highest_leakag:2,hold:10,home:[8,9,10],how:[2,10],howev:8,hyper:4,hypothesi:2,i:[2,6,8,9,10],ignor:10,ill:8,im:6,imag:2,implement:[8,9,10],impresison:9,includ:[0,2,9],incom:9,index:0,indic:[2,6,9,10],individu:[9,10],ine:4,inequ:10,inf:2,infer:1,inferencedata:[6,10],inform:[0,2,6,8],informaiton:9,initi:10,input:0,input_inferencedata:[6,9,10],input_spec:[8,9,10],inspect:10,instanc:10,insuffici:8,interest:[6,8,10],interfac:[9,10],intgener:2,intlist:2,introduc:10,invalid:[8,9],is_hyper_param:4,isol:8,iter:[8,9,10],its:2,jitter:10,job:[8,9,10],join:[8,9,10],just:8,kl:0,know:[0,8,10],knowledg:[0,9],known:8,kwarg:[8,9],l:2,label:10,lam:4,lambda:[2,4,8,9],laplac:[2,9],laplacian:9,larg:[2,10],larger:[8,9,10],last:9,lastli:9,leakag:[0,2,9],legend:10,len:[8,9],length:2,leq:9,less:10,let:10,level:8,lib:[8,9,10],librai:9,librari:[0,10],like:[8,10],line:[8,9,10],list:[2,4,10],live:8,ln:6,load:[2,9],load_from_fil:2,loc:10,local:[8,9,10],locat:2,log2:6,logarithm:6,look:10,low:[2,8,9,10],lower:[2,4,10],lt:10,m:2,mai:[8,9,10],male:8,mani:[2,9,10],manner:9,mark:10,marri:9,marriag:9,masachusset:8,mass:9,match:[9,10],mathbb:[8,10],mathit:9,matplotlib:[8,9,10],matrix:8,max:9,max_subplot:10,maxim:0,mean:[4,9,10],measur:[0,1,9,10],medic:8,merg:8,method:[2,8,9,10],metric:1,metropoli:[8,9],mi_sklearn:[6,9,10],might:10,mimic:2,min:[0,9],minimum:0,minimum_coverag:2,minu:10,mode:10,model:[8,9,10],modul:0,more:[8,9,10],most:[8,10],move:[9,10],mu:[2,4,8,9,10],multipl:2,multiprocess:[8,9,10],must:[0,6,9],mutual:[0,2,6],mutual_info_regress:6,mutual_inform:[1,2],n:[2,4,8,9,10],n_neigh:[6,9],n_rv:9,name:[2,4,8,9,10],names_db:8,names_oth:8,nan:[8,9],natur:[6,9],necessari:9,necessarili:10,need:[2,10],neighbour:6,next:[2,9,10],nois:9,nomin:[8,9],non:9,none:4,normal:[2,4,8,9,10],note:[6,8,9,10],notebook:10,now:8,nowadai:8,np:[8,9,10],npmodul:[8,9],nputil:[8,9],nu:2,num_el:[4,8,9,10],num_sampl:[8,9],number:[2,4,6,8,9,10],numer:[8,9],numper:2,numpi:[8,9,10],nut:[8,9,10],o:10,object:[2,6,8,9,10],observ:[8,9],obtain:10,old:[8,9],one:[8,10],onli:[8,10],opendp:0,option:8,orang:10,order:[0,9],os:[8,9,10],other:[0,8,9,10],otherwis:10,our:10,outcom:10,output:[8,9,10],output_typ:[8,9,10],over:9,p:[2,4,8,9,10],packag:[8,9,10],page:0,pair:2,panda:9,paper:10,paramet:[2,4,6,8,9,10],pardo:[8,9,10],part:[8,9],particular:2,patch_artist:9,path:[2,8,9,10],pd:9,peopl:8,perform:[0,8,9,10],perspect:8,pickl:2,placehold:8,plausibl:10,plot:[2,8,9,10],plot_dist:10,plot_mutual_bar:2,plot_mutual_inform:2,plot_posterior:10,plot_util:10,plt:[8,9,10],pm:10,pmf:2,point:[8,9],point_estim:10,poisson:2,pose:10,possibl:[2,9,10],possible_dist:2,posterior:[8,9],posterior_alice_ag:10,precis:10,predic:10,present:8,previous:[9,10],print:[2,8,10],prior:[0,9],prior_alice_ag:10,priv:2,privaci:[0,10],privacy_usag:9,privug:[9,10],privugg:[2,8],probabilist:2,probabl:[0,2,4,8,9],problem:8,program:0,protect:[0,9],publicli:[0,10],purpos:[8,10],put:10,pv:[8,9,10],py:[8,9,10],pymc3:[2,8,9,10],pymc3_dist:4,pyplot:[8,9,10],python3:[8,9,10],python:[0,9,10],quantifi:[8,9,10],queri:[0,8],quit:2,r:[0,10],race:9,random:[2,4,9],rang:[2,8,9],rcparam:10,re:8,real:8,realiti:8,recal:10,record:9,ref_val:10,refin:9,relat:9,releas:[9,10],remain:9,remak:9,remark:[8,9],rememb:10,remove_nam:8,repres:2,requir:10,respect:2,rest:8,result:[6,9],return_model:[8,10],rhat:[8,9],risk:[0,10],row:8,runtimewarn:[8,9],rv:[2,4],s:[2,4,10],sai:10,same:[8,9],sampl:[6,8,9,10],sample_prior_predict:10,sampler:[8,9,10],save:2,save_to_fil:2,scalar:10,scipi:10,scipy_dist:4,sd:10,search:0,second:[8,9,10],secret:6,section:10,see:[8,9,10],seem:9,sens:2,sensit:8,serv:8,set:[9,10],sex:9,shape:2,she:[8,10],shift:[2,10],should:[2,10],show:[8,9,10],showflier:9,showmean:9,sight:10,sigma:2,signatur:10,simpl:10,simplefilt:10,simpli:[8,9],simplif:8,simul:2,simulationmetr:2,sinc:[2,8,9],singl:[2,8,10],site:[8,9,10],size:[2,9,10],sklearn:[6,10],slice:[8,9,10],smaller:[8,9,10],smartnois:9,sn:9,snippet:9,so:8,some:[8,9,10],someth:2,sourc:[2,4,6],spec:[8,10],specif:[0,2],specifi:[4,9,10],standard:[4,9],start:0,stat:[8,9],statist:[8,9],statu:9,std:[4,8,9,10],step:[8,9,10],still:8,store:2,str:[2,10],strength:10,string:[2,4,6],structur:9,student_t:2,studentt:2,studi:[0,9],sum:[8,10],support:[2,9,10],sweeni:8,sy:[8,9,10],synthesi:0,t:8,take:[0,2,8,9,10],tell:0,temp:9,temp_fil:9,tensor:[8,9,10],test:2,than:[2,8,9,10],theano:[8,9,10],thei:[9,10],them:[2,8,9],theoret:10,thi:[0,4,6,8,9,10],think:10,those:[8,9],though:[8,9],titl:[8,9,10],to_csv:9,to_float:9,togeth:10,took:[8,9,10],tool:10,total:[8,9,10],total_n:8,total_sampl:8,toward:10,trace:[2,6,8,9,10],trace_attr:[8,9],trace_length:9,transform:1,treat:9,trial:4,truncated_norm:2,truncatednorm:2,tune:[8,9,10],tupl:2,tutori:[8,9,10],two:[6,8],type:[2,4,6,8,9,10],u:2,un:2,unaccept:8,uncertainti:10,under:[9,10],uniform:[2,4,9],uniformli:8,unnatur:9,unrealist:8,up:10,upper:[2,4],us:[0,2,4,6,8,10],user:0,userwarn:10,val:4,valu:[2,4,9,10],var_nam:[6,9,10],variabl:[2,4,6,8,9,10],varianc:10,varieti:0,vector:9,verbos:2,veri:[9,10],verifi:10,vert:9,victim:8,wa:[8,10],wai:9,want:10,warn:[8,9,10],we:[8,9,10],well:8,were:[8,10],what:[0,10],when:8,where:[8,9],whether:[6,8,10],which:[2,8,9,10],who:8,whose:10,wide:0,within:2,within_chain_vari:[8,9],without:8,word:[0,10],work:9,wrap:[9,10],x:2,xarrai:[8,9],xlabel:[8,9,10],xlim:8,xtick:9,xxx_other:8,y1:8,y2:8,y3:8,y:[2,9],year:[8,9],yet:8,ylabel:9,you:[8,10],zip:8,zips_db:8,zips_oth:8},titles:["Welcome to privugger\u2019s documentation!","privugger package","Attacker Generation","Privugger Datastructures","Privugger Distributions","Privugger Inference","Privugger Measures","Privugger Transformer","Governor case study","Example of using Privugger on OpenDP","Getting started with privugger"],titleterms:{"case":8,"na\u00efv":8,about:10,alic:8,analysi:[8,9,10],analyz:10,anonym:8,attack:[2,10],attribut:8,base:8,content:1,continu:4,data_structur:3,dataset:9,datastructur:3,discret:4,discuss:8,distribut:[2,4,10],document:0,exampl:[0,9],flow:10,gener:2,get:10,governor:8,histogram:8,how:8,identifi:8,indic:0,infer:[5,8,9,10],inform:[9,10],input:[8,9,10],knowledg:10,mani:8,measur:6,metric:[2,10],modul:[1,2,3,4,5],mutual:[9,10],mutual_inform:6,observ:10,opendp:9,packag:1,posterior:10,prior:10,privaci:[8,9],privugg:[0,1,3,4,5,6,7,9,10],probabl:10,program:[8,9,10],qif:10,quantit:10,queri:10,record:8,remov:8,result:8,risk:[8,9],s:[0,8],share:8,specif:[8,9,10],standard:10,start:10,statist:10,studi:8,subpackag:1,tabl:0,transform:7,tutori:0,uniqu:8,us:9,valu:8,visual:10,vs:10,welcom:0}})