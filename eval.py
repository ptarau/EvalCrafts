import glob
import sys
import os

import rouge_stats as rs
import key_stats as ks
from itertools import islice

import textcrafts
from textcrafts import deepRank as dr
from textcrafts.sim import *

WITH_DOCTALK=0

from doctalk.talk import Talker, nice_keys


def customGraphMaker() : # CHOICE OF PARSER TOOLKIT
  return dr.GraphMaker(params=dr.params)
  #return dr.GraphMaker(api_classname=CoreNLP_API)

# sets max s number of documents to be processed, all if None
max_docs = None
# resource directories, for production and testing at small scale
prod_mode=False
# shows moving averages if on
trace_mode=False
# if true abstracts are not trimmed out from documents
with_full_text = False
# number of keyphrases and summary sentences
wk,sk=9,10
#wk,sk=8,9
#wk,sk=5,9
#wk,sk=5,10

if prod_mode :
  data_dir='dataset/Krapivin2009/'
  show_errors=False
else :
  data_dir='dataset/small/'
  show_errors = True
  
doc_dir=data_dir+'docsutf8/'
keys_dir=data_dir+'keys/'
abs_dir=data_dir+'abs/'
all_doc_files = sorted(glob.glob(doc_dir+"*.txt"))

# END OF PARAMS

if max_docs :
  doc_files=list(islice(all_doc_files,max_docs))
else :
  doc_files=all_doc_files

if prod_mode :
  out_abs_dir  = "out/abs/"
  out_keys_dir = "out/keys/"
else :
  out_abs_dir  = "test/abs/"
  out_keys_dir = "test/keys/"

# clean output directories
def clean_all() :
  clean_path(out_abs_dir)
  clean_path(out_keys_dir)

# clean files at given directory path 
def clean_path(path) :
  os.makedirs(path,exist_ok=True)

  files = glob.glob(path+"/*")
  for f in files:
    os.remove(f)
   
# extract triple (title,abstract,body) with refs trimmed out
def disect_doc(doc_file) :
  title=[]
  abstract=[]
  body=[]
  mode=None
  with open(doc_file) as f:
    for line in f:
      if line.startswith('--T')   : mode='TITLE'
      elif line.startswith('--A') : mode ='ABS'
      elif line.startswith('--B') : mode = 'BODY'
      elif line.startswith('--R'): mode = 'DONE'
      else :
        if   mode=='TITLE': title.append(line.strip()+' ')
        elif mode=='ABS'  : abstract.append(line.strip()+' ')
        elif mode=='BODY' : body.append(line.strip()+' ')
        elif mode=='DONE' : break
  return {'TITLE':title,'ABSTRACT':abstract,'BODY':body}

# process string text give word count,sentence count and filter
def runWithText(text,wk,sk,filter) :
  gm=customGraphMaker()
  gm.digest(text)
  keys= gm.bestWords(wk)
  sents=[s for (_,s) in gm.bestSentences(sk)]
  #keys_text=interleave_with('\n','\n',keys)
  #sents_text=interleave_with('\n','\n',sents)
  #return (keys_text,sents_text)
  nk=gm.nxgraph.number_of_nodes()
  vk=gm.nxgraph.number_of_edges()
  return (keys,sents,nk,vk)

def runWithTextAlt(text,wk,sk,filter) :

  talker=Talker(from_text=text)
  ranked_sents,keys=talker.extract_content(sk,wk)

  def clean_sents():
    for r, s, ws in ranked_sents:
      yield ws

  #print('!!!KEYS',keys)
  #print('!!!SENT',list(clean_sents()))
  keys=nice_keys(keys)
  return (keys,clean_sents(),talker.g.number_of_nodes(),talker.g.number_of_edges())


#  extract the gold standard abstracts from dataset  
def fill_out_abs() :
   for doc_file in doc_files :
     d=disect_doc(doc_file)
     abstract=d['ABSTRACT']
     text=''.join(abstract)
     abs_file=abs_dir+dr.path2fname(doc_file)
     print('abstract extraced to: ',abs_file)
     string2file(abs_file,text)

     
# turns a sequence/generator into a file, one line per item yield     
def seq2file(fname,seq) :
  xs=map(str,seq)
  ys=interleave_with('\n','\n',xs)
  text=''.join(ys)
  string2file(fname,text)

# turns a file into a (string) generator yielding each of its lines
def file2seq(fname) :
   with open(fname,'r') as f :
     for l in f : yield l.strip()

# turns a string into given file
def string2file(fname,text) :
  with open(fname,'w') as f :  
    f.write(text)

# turns content of file into a string
def file2string(fname) :
  with open(fname,'r') as f :
    s = f.read()
    return s.replace('-',' ')

# interleaves list with separator
def interleave(sep,xs) :
  return interleave_with(sep,None,xs)
  
def interleave_with(sep,end,xs) :
  def gen() :
    first=True
    for x in xs : 
      if not first : yield sep
      yield x
      first=False
    if end : yield(end)
      
  return ''.join(gen())

def process_file(path_file,full,wk,sk) :
  doc_file = dr.path2fname(path_file)
  d = disect_doc(path_file)
  title = d['TITLE']
  abstract = d['ABSTRACT']
  body = d['BODY']
  text_no_abs = ''.join(title + [' '] + body)

  if full:
    text = ''.join(title + [' '] + abstract + [' '] + body)
  else:
    text = ''.join(title + [' '] + body)

  if WITH_DOCTALK :
    (keys, xss, nk, ek) = runWithTextAlt(text, wk, sk, dr.isWord)
  else:
    (keys, xss, nk, ek) = runWithText(text, wk, sk, dr.isWord)

  print(doc_file, 'nodes:', nk, 'edges:', ek)  # ,title)
  exabs = map(lambda x: interleave(' ', x), xss)
  kf = out_keys_dir + doc_file
  af = out_abs_dir + doc_file
  seq2file(kf, keys)
  seq2file(af, exabs)


# extracts keys and abstacts from resource directory  
def extract_keys_and_abs(full,wk,sk,show_errors=show_errors) :
  clean_all()
  for path_file in doc_files :
    if show_errors:
      process_file(path_file, full, wk, sk)
    else:
      try :
        process_file(path_file, full, wk, sk)
      except :
        print('*** FAILING on:',doc_file,'ERROR:',sys.exc_info()[0])

# apply Python base rouge to abstracts from given directory
def eval_with_rouge(i) :
  f=[]
  p=[]
  r=[]  
  for doc_file in doc_files : 
    fname=dr.path2fname(doc_file)
    ref_name=abs_dir+fname
    abs_name=out_abs_dir+fname
    #if trace_mode : print(fname)
    gold=file2string(ref_name)   
    silver=file2string(abs_name)
    k=0
    for res in rs.rstat(silver,gold) :
      if k==i:    
        d=res[0]
      
        px=d['p'][0]
        rx=d['r'][0]
        fx=d['f'][0]
    
        p.append(px)
        r.append(rx)
        f.append(fx)
        
      elif k>i : break
      k+=1
    if trace_mode : print('  ABS ROUGE MOV. AVG',i,fname,avg(p),avg(r),avg(f))
  rouge_name=(1,2,'l','w')  
  print ("ABS ROUGE",rouge_name[i],':',avg(p),avg(r),avg(f))

# our own 
def eval_abs() :
  f=[]
  p=[]
  r=[]  
  for doc_file in doc_files : 
    fname=dr.path2fname(doc_file)
    ref_name=abs_dir+fname
    abs_name=out_abs_dir+fname
    #if trace_mode : print(fname)
    gold=file2string(ref_name)
    silver=file2string(abs_name)
    #print(gold)
    #print(silver)
    d=ks.kstat(silver,gold)
    if not d :
      print('FAILING on',fname)
      continue
    if trace_mode: print('  ABS SCORE:',d)
    px=d['p']
    rx=d['r']
    fx=d['f']
    if px and rx and fx :
      p.append(px)
      r.append(rx)
      f.append(fx)
    if trace_mode : print('  ABS MOV. AVG',fname,avg(p),avg(r),avg(f))
  print ("ABS SCORES  :",avg(p),avg(r),avg(f))

  
# 0.22434732994628803 0.24271988542882067 0.22280040709372084
def eval_keys() :
  f=[]
  p=[]
  r=[]  
  for doc_file in doc_files : 
    fname=dr.path2fname(doc_file)
    ref_name=keys_dir+fname
    keys_name=out_keys_dir+fname
    #if trace_mode : print(fname)
    gold=file2string(txt2key(ref_name))   
    silver=file2string(keys_name)
    #print(gold)
    #print(silver)
    d=ks.kstat(silver,gold)
    if not d :
      print('FAILING on',fname)
      print('SILVER',silver)
      print('GOLD',gold)
      continue
    if trace_mode : print('  KEYS',d)
    px=d['p']
    rx=d['r']
    fx=d['f']
    p.append(px)
    r.append(rx)
    f.append(fx)
    #if trace_mode : print('  KEYS . AVG:',fname,avg(p),avg(r),avg(f))
  print('KEYS SCORES :',avg(p),avg(r),avg(f))
  
  
def txt2key(fname) :
  return fname.replace('.txt','.key')
    
def avg(xs) :
  s=sum(xs)
  l=len(xs)
  if 0==l : return None
  return s/l  

######### main evaluator #############
def go() :

  #fill_out_abs


  def showParams(p=dr.params) :
    if WITH_DOCTALK : print('WITH_DOCTALK')
    else: print('WITH_TEXTCRAFT')
    print(
          'wk',wk,'sk',sk,'\n'
          'with_full_text = ',with_full_text,'\n',
          'prod_mode = ' ,prod_mode,'\n',
          'max_docs = ',max_docs,'\n',
          #'noun_defs = ',p.noun_defs,'\n',
          #'all_recs =',p.all_recs,'\n'
          )

  print("STARTING")
  showParams()
  extract_keys_and_abs(with_full_text, wk, sk)
  print("EXTRACTED KEYS AND ABSTRACTS")
  eval_keys()
  eval_abs()
  eval_with_rouge(0)  # 1
  eval_with_rouge(1)  # 2
  eval_with_rouge(2)  # l
  eval_with_rouge(3)  # w
  print('DONE')
  showParams()

if __name__ == '__main__' :
  pass
  #go()

'''
sqrt
KEYS SCORES : 0.27416666666666667 0.34694444444444444 0.29275340952551887
ABS SCORES  : 0.36118382771545704 0.5042334736635918 0.41332132103435465
ABS ROUGE 1 : 0.3933145118929155 0.5184928165336313 0.43482299877528385
ABS ROUGE 2 : 0.16357039735115525 0.232778796923158 0.1858206751162993
ABS ROUGE l : 0.34487947667156577 0.442321878246675 0.3792731992052987
ABS ROUGE w : 0.1964525340859074 0.09936084572373194 0.12626315636206803

log
KEYS SCORES : 0.27416666666666667 0.34694444444444444 0.29275340952551887
ABS SCORES  : 0.37839101818507703 0.5385774853321073 0.4364521548054726
ABS ROUGE 1 : 0.4046478727360066 0.5505292131459584 0.45296490614873336
ABS ROUGE 2 : 0.19829718998915621 0.2972863472526791 0.2295444270832006
ABS ROUGE l : 0.36403723150031414 0.4788527681008241 0.40464878815939126
ABS ROUGE w : 0.21642668012136568 0.11444788050837104 0.1430266946676839


MAIN:

EXTRACTED KEYS AND ABSTRACTS
KEYS SCORES : 0.29503968253968255 0.3219444444444445 0.2959468380440248
ABS SCORES  : 0.33452712584341465 0.4740477680258356 0.3845643390518048
ABS ROUGE 1 : 0.3760798082044753 0.5094655020968573 0.4181257944004031
ABS ROUGE 2 : 0.12405609245796925 0.1875968420204949 0.1431943359529297
ABS ROUGE l : 0.3215833598499711 0.4227713509162004 0.35590257892454036
ABS ROUGE w : 0.16300628670324277 0.0864869054013174 0.10753312632582054
DONE
wk 9 sk 10 
with_full_text =  True 

---------------------

EXTRACTED KEYS AND ABSTRACTS
KEYS SCORES : 0.32063492063492066 0.34734126984126984 0.3178978701037525
ABS SCORES  : 0.30416748129317933 0.4112751156488142 0.34242633414995327
ABS ROUGE 1 : 0.35535878093259593 0.4603085331098845 0.3873763551896513
ABS ROUGE 2 : 0.07602657775654001 0.09304083704093895 0.08078371367464267
ABS ROUGE l : 0.2880671913696598 0.36169737849970984 0.3122201684391084
ABS ROUGE w : 0.1322837472128466 0.06307627558309192 0.08143434144704587
DONE
WITH_TEXTCRAFT
wk 9 sk 10 
with_full_text =  False 
 prod_mode =  False 
 max_docs =  None 
 
EXTRACTED KEYS AND ABSTRACTS
KEYS SCORES : 0.3733333333333334 0.26896825396825397 0.29361042466305626
ABS SCORES  : 0.30416748129317933 0.4112751156488142 0.34242633414995327
ABS ROUGE 1 : 0.35535878093259593 0.4603085331098845 0.3873763551896513
ABS ROUGE 2 : 0.07602657775654001 0.09304083704093895 0.08078371367464267
ABS ROUGE l : 0.2880671913696598 0.36169737849970984 0.3122201684391084
ABS ROUGE w : 0.1322837472128466 0.06307627558309192 0.08143434144704587
DONE
WITH_TEXTCRAFT
wk 5 sk 10 
with_full_text =  False 
 prod_mode =  False 
 max_docs =  None 
 
ALT =====================================

KEYS SCORES : 0.2888888888888889 0.3600396825396825 0.3054839037064101
ABS SCORES  : 0.32232775575193784 0.45299073003787027 0.3663694775557215
ABS ROUGE 1 : 0.3876319785001836 0.5018374555311037 0.41952966445678525
ABS ROUGE 2 : 0.10116519845733665 0.12818901937380384 0.10823287672321191
ABS ROUGE l : 0.32237808748837565 0.4096058696356243 0.35004520068797124
ABS ROUGE w : 0.15479348038132082 0.07611683764667457 0.09673993899837299
DONE
wk 9 sk 10 
with_full_text =  False 


EXTRACTED KEYS AND ABSTRACTS
KEYS SCORES : 0.32591269841269843 0.395515873015873 0.3467373701813061
ABS SCORES  : 0.32232775575193784 0.45299073003787027 0.3663694775557215
ABS ROUGE 1 : 0.3876319785001836 0.5018374555311037 0.41952966445678525
ABS ROUGE 2 : 0.10116519845733665 0.12818901937380384 0.10823287672321191
ABS ROUGE l : 0.32237808748837565 0.4096058696356243 0.35004520068797124
ABS ROUGE w : 0.15479348038132082 0.07611683764667457 0.09673993899837299
DONE
WITH_DOCTALK
wk 5 sk 10 
with_full_text =  False 

EXTRACTED KEYS AND ABSTRACTS
KEYS SCORES : 0.3364285714285714 0.395515873015873 0.34962635317898483
ABS SCORES  : 0.3851968760343332 0.5352845542945579 0.4361280011212781
ABS ROUGE 1 : 0.43139969007124035 0.5591830800838007 0.4689422470011375
ABS ROUGE 2 : 0.21269730727127412 0.2676123452550708 0.2279407602659072
ABS ROUGE l : 0.3912572762904009 0.4922857385622622 0.4241039456958681
ABS ROUGE w : 0.22618450593531633 0.10608733998121511 0.1372011320702345
DONE
WITH_DOCTALK
wk 5 sk 10 
with_full_text =  True 
 prod_mode =  False 
 max_docs =  None 
 
--------------------------


KEYS SCORES : 0.30000000000000004 0.37115079365079356 0.3165950148175213
ABS SCORES  : 0.3851968760343332 0.5352845542945579 0.4361280011212781
ABS ROUGE 1 : 0.43139969007124035 0.5591830800838007 0.4689422470011375
ABS ROUGE 2 : 0.21269730727127412 0.2676123452550708 0.2279407602659072
ABS ROUGE l : 0.3912572762904009 0.4922857385622622 0.4241039456958681
ABS ROUGE w : 0.22618450593531633 0.10608733998121511 0.1372011320702345
DONE
wk 9 sk 10 
with_full_text =  True 
'''
