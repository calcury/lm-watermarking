# coding=utf-8
"""Colab-friendly robustness evaluation following arXiv:2301.10226v4.

The paper's central budget is epsilon*T token edits.  This script evaluates
multiple generated samples, writes per-sample records, and writes an aggregate
TPR/FPR/FNR/AUC evaluation matrix.  A T5/FLAN attacker can be downloaded with
--attacker_model; without it, model-dependent attacks are explicitly skipped.
"""
import argparse, csv, os, random, re, time
from dataclasses import dataclass, asdict
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector

EPSILONS = (0.0, .1, .3, .5, .7)
ZW = ("\u200b", "\u200c", "\u200d", "\ufeff")
EMOJI = ("🙂", "🔍", "✨", "🌊", "📚")
HOMO = {"a":"а","c":"с","e":"е","o":"о","p":"р","s":"ѕ","x":"х","y":"у","A":"А","B":"В","C":"С","E":"Е","H":"Н","K":"К","M":"М","O":"О","P":"Р","T":"Т","X":"Х","Y":"У"}

@dataclass
class Record:
    attack: str; epsilon: float; source: str; sample: int; status: str
    z_score: Optional[float]=None; green_fraction: Optional[float]=None
    token_count: Optional[int]=None; detected: Optional[bool]=None
    perplexity: Optional[float]=None; token_edit_ratio: Optional[float]=None
    requested_edits: Optional[int]=None; actual_edits: Optional[int]=None
    elapsed_seconds: Optional[float]=None; note: str=""

def ids(text, tok): return tok(text, add_special_tokens=False)["input_ids"]
def budget(text, tok, eps): return min(len(ids(text,tok)), round(eps*len(ids(text,tok))))
def edit_ratio(a,b,tok):
    x,y=ids(a,tok),ids(b,tok); return 1-sum(i==j for i,j in zip(x,y))/max(1,max(len(x),len(y)))
def R(seed): return random.Random(seed)

def discreet(text, eps, tok, seed):
    chars=list(text); n=min(len(chars),budget(text,tok,eps)); r=R(seed)
    for i in r.sample(range(len(chars)),n) if n else []:
        if chars[i].isalpha(): chars[i]=chars[i].lower() if chars[i].isupper() else chars[i]
        elif chars[i]==' ': chars[i]='  '
    return ''.join(chars)

def tokenization(text,eps,tok,seed):
    ms=list(re.finditer(r'[A-Za-z]{4,}',text)); n=min(len(ms),budget(text,tok,eps)); r=R(seed)
    cuts={m.start()+max(1,len(m.group())//2) for m in (r.sample(ms,n) if n else [])}
    return ''.join((' ' if i in cuts else '')+c for i,c in enumerate(text))

def homoglyph(text,eps,tok,seed):
    chars=list(text); pos=[i for i,c in enumerate(chars) if c in HOMO]; n=min(len(pos),budget(text,tok,eps))
    for i in R(seed).sample(pos,n) if n else []: chars[i]=HOMO[chars[i]]
    return ''.join(chars)

def zero_width(text,eps,tok,seed):
    n=min(len(text),budget(text,tok,eps)); pos=set(R(seed).sample(range(len(text)),n)) if n else set(); r=R(seed+1)
    return ''.join(c+(r.choice(ZW) if i in pos else '') for i,c in enumerate(text))

def emoji(text,eps,tok,seed):
    ws=text.split(' '); n=min(len(ws),budget(text,tok,eps)); r=R(seed)
    for i in r.sample(range(len(ws)),n) if n else []: ws[i]+=r.choice(EMOJI)
    return ' '.join(ws)

def insertion(text,eps,tok,seed):
    ws=text.split(' '); r=R(seed)
    for _ in range(budget(text,tok,eps)): ws.insert(r.randrange(len(ws)+1),r.choice(('however','indeed','notably','also')))
    return ' '.join(ws)

def paraphrase(text,eps,tok,seed):
    ws=text.split(); n=min(len(ws)//2,budget(text,tok,eps)); r=R(seed)
    for _ in range(n):
        if len(ws)>3:
            i=r.randrange(len(ws)-1); ws[i],ws[i+1]=ws[i+1],ws[i]
    return ' '.join(ws)

def rewrite(text,model,tok,device,seed):
    prompt='Paraphrase this text, preserve its meaning, return only the paraphrase:\n'+text
    x=tok(prompt,return_tensors='pt',truncation=True,max_length=1024).to(device); torch.manual_seed(seed)
    source_len=len(ids(text,tok))
    y=model.generate(**x,min_new_tokens=max(16, int(source_len*.6)), max_new_tokens=max(32, int(source_len*1.4)), do_sample=True,temperature=.8,top_p=.95)
    result=tok.decode(y[0],skip_special_tokens=True).strip()
    return result or text

def t5_span(text,model,tok,device,eps,base_tok,seed):
    ws=text.split(); requested=min(len(ws),budget(text,base_tok,eps)); actual=0; r=R(seed)
    for _ in range(requested):
        if not ws: break
        i=r.randrange(len(ws)); old=ws[i]; masked=' '.join(ws[:i]+['<extra_id_0>']+ws[i+1:])
        x=tok(masked,return_tensors='pt',truncation=True,max_length=1024).to(device)
        ys=model.generate(**x,max_new_tokens=12,num_beams=50,num_return_sequences=20)
        replaced=False
        for y in ys:
            s=tok.decode(y,skip_special_tokens=False)
            m=re.search(r'<extra_id_0>\s*(.*?)\s*(?:<extra_id_1>|</s>|$)',s,re.S)
            if m:
                candidate=m.group(1).replace('<pad>','').strip()
                if candidate and candidate != old and '<extra_id_' not in candidate:
                    ws[i]=candidate; actual += 1; replaced=True; break
        if not replaced:
            continue
    return ' '.join(ws), requested, actual

def score(text,args,tok,device):
    try:
        d=WatermarkDetector(vocab=list(tok.get_vocab().values()),gamma=args.gamma,seeding_scheme=args.seeding_scheme,device=device,tokenizer=tok,z_threshold=args.threshold,normalizers=[],ignore_repeated_bigrams=False,select_green_tokens=True).detect(text)
        return d,''
    except Exception as e: return None,f'{type(e).__name__}: {e}'

def ppl(text,model,tok,device):
    if model is None:return None
    x=tok(text,return_tensors='pt',truncation=True,max_length=1024).to(device)
    with torch.no_grad(): return float(torch.exp(model(**x,labels=x['input_ids']).loss).cpu())

def auc(labels,scores):
    p=sum(labels); n=len(labels)-p
    if not p or not n:return None
    ranks=sorted(zip(scores,labels)); s=sum(i+1 for i,(_,y) in enumerate(ranks) if y)
    return (s-p*(p+1)/2)/(p*n)

def load_model(name, device):
    if not name:return None,None
    tok=AutoTokenizer.from_pretrained(name)
    return AutoModelForCausalLM.from_pretrained(name).to(device).eval(),tok

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default=os.getenv('WATERMARK_MODEL','facebook/opt-125m')); p.add_argument('--attacker_model',default=os.getenv('ATTACKER_MODEL',''))
    p.add_argument('--samples',type=int,default=5); p.add_argument('--tokens',type=int,default=200)
    p.add_argument('--quality_model',default=os.getenv('QUALITY_MODEL',''), help='Independent causal LM for paper-style PPL; defaults to the target model.')
    p.add_argument('--quality',action='store_true', help='Compute perplexity using --quality_model or the target model.')
    p.add_argument('--output_csv',default='attack_records.csv'); p.add_argument('--matrix_csv',default='evaluation_matrix.csv')
    p.add_argument('--attacks',nargs='+',default=['paraphrasing','discreet_alterations','tokenization','homoglyph','zero_width','generative','emoji','insertion']); a=p.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'; target_tok=AutoTokenizer.from_pretrained(a.model); target=AutoModelForCausalLM.from_pretrained(a.model).to(device).eval(); attacker,attok=load_model(a.attacker_model,device); quality=None
    if a.quality:
        quality_tok=AutoTokenizer.from_pretrained(a.quality_model or a.model)
        quality=target if (a.quality_model or a.model)==a.model else AutoModelForCausalLM.from_pretrained(a.quality_model).to(device).eval()
    else:
        quality_tok=target_tok
    args=type('A',(),dict(gamma=.25,delta=2.,seeding_scheme='simple_1',threshold=4.))(); prompt='The diamondback terrapin is a species of turtle native to coastal marshes. It has a distinctive shell and lives in brackish water. The species is'
    records=[]
    for sample in range(a.samples):
        torch.manual_seed(123+sample); x=target_tok(prompt,return_tensors='pt').to(device); out=target.generate(**x,max_new_tokens=a.tokens,do_sample=True,temperature=.7); plain=target_tok.decode(out[0][x['input_ids'].shape[-1]:],skip_special_tokens=True)
        proc=WatermarkLogitsProcessor(vocab=list(target_tok.get_vocab().values()),gamma=.25,delta=2.,seeding_scheme='simple_1',select_green_tokens=True); torch.manual_seed(123+sample); out=target.generate(**x,max_new_tokens=a.tokens,do_sample=True,temperature=.7,logits_processor=LogitsProcessorList([proc])); marked=target_tok.decode(out[0][x['input_ids'].shape[-1]:],skip_special_tokens=True)
        for attack in a.attacks:
            for eps in EPSILONS:
                for source,original in (('unwatermarked',plain),('watermarked',marked)):
                    t=time.perf_counter(); requested=budget(original,target_tok,eps); actual=0
                    try:
                        if eps == 0.0:
                            attacked=original
                        elif attack=='generative':
                            if not attacker: raise RuntimeError('requires --attacker_model')
                            attacked=rewrite(original,attacker,attok,device,sample)
                            if len(ids(attacked,target_tok)) < max(3, int(.5*len(ids(original,target_tok)))):
                                raise RuntimeError('invalid_short_output')
                            actual=max(1, round(edit_ratio(original,attacked,target_tok)*len(ids(original,target_tok))))
                        elif attack in ('lm_span_replacement','t5_span_replacement'):
                            if not attacker: raise RuntimeError('requires --attacker_model')
                            attacked,requested,actual=t5_span(original,attacker,attok,device,eps,target_tok,sample)
                        else:
                            fn={'paraphrasing':paraphrase,'discreet_alterations':discreet,'tokenization':tokenization,'homoglyph':homoglyph,'zero_width':zero_width,'emoji':emoji,'insertion':insertion}[attack]; attacked=fn(original,eps,target_tok,sample)
                            actual=round(edit_ratio(original,attacked,target_tok)*len(ids(original,target_tok)))
                        d,note=score(attacked,args,target_tok,device)
                        if d is None: raise RuntimeError(note)
                        records.append(Record(attack,eps,source,sample,'ok',d['z_score'],d['green_fraction'],d['num_tokens_scored'],d['prediction'],ppl(attacked,quality,quality_tok,device),edit_ratio(original,attacked,target_tok),requested,actual,time.perf_counter()-t))
                    except Exception as e: records.append(Record(attack,eps,source,sample,'skipped',requested_edits=requested,elapsed_seconds=time.perf_counter()-t,note=str(e)))
    fields=list(asdict(records[0]).keys())
    with open(a.output_csv,'w',newline='',encoding='utf8') as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(asdict(r) for r in records)
    with open(a.matrix_csv,'w',newline='',encoding='utf8') as f:
        fs=['attack','epsilon','samples','valid_watermarked','valid_unwatermarked','tpr','fpr','fnr','auc','mean_z_watermarked','mean_z_unwatermarked','mean_green_watermarked','mean_ppl','std_ppl'];w=csv.DictWriter(f,fieldnames=fs);w.writeheader()
        for attack in a.attacks:
            for eps in EPSILONS:
                rows=[r for r in records if r.attack==attack and r.epsilon==eps and r.status=='ok']; wm=[r for r in rows if r.source=='watermarked']; uw=[r for r in rows if r.source=='unwatermarked']; labels=[r.source=='watermarked' for r in rows]; scores=[r.z_score for r in rows]; tpr=sum(r.detected for r in wm)/len(wm) if wm else None; fpr=sum(r.detected for r in uw)/len(uw) if uw else None
                ppls=[r.perplexity for r in rows if r.perplexity is not None]
                mean_ppl=sum(ppls)/len(ppls) if ppls else None
                std_ppl=(sum((x-mean_ppl)**2 for x in ppls)/len(ppls))**.5 if ppls else None
                w.writerow(dict(attack=attack,epsilon=eps,samples=len(rows),valid_watermarked=len(wm),valid_unwatermarked=len(uw),tpr=tpr,fpr=fpr,fnr=1-tpr if tpr is not None else None,auc=auc(labels,scores) if rows else None,mean_z_watermarked=sum(r.z_score for r in wm)/len(wm) if wm else None,mean_z_unwatermarked=sum(r.z_score for r in uw)/len(uw) if uw else None,mean_green_watermarked=sum(r.green_fraction for r in wm)/len(wm) if wm else None,mean_ppl=mean_ppl,std_ppl=std_ppl))
    print(f'Wrote {len(records)} records to {a.output_csv}');print(f'Wrote evaluation matrix to {a.matrix_csv}')
if __name__=='__main__': main()
