#!/usr/bin/env python3
"""Stronger reproducible null-model tests for the Dan Panorama RTM case.

The program verifies the 16 stated closures and performs four tests:
1. Full-event Monte Carlo: randomizes dates, room/floor, names, ages and phones,
   then reconstructs all text-derived values and applies the frozen formulas.
2. Exact room/date grid: exhaustively tests 450,000 plausible room/date sequences.
3. Exact semantic-role permutation: shuffles the observed name/date/age roles.
4. Free-relation density test: gives random matched sets broad freedom to find any
   short arithmetic relations, rather than only the stated formulas.

Reproduction command used for the accompanying report:
    python dan_panorama_stronger_tests.py \
        --full-trials 100000000 --density-generated 1000000

Requires: Python 3, numpy, numba.
"""
from __future__ import annotations

import argparse
import calendar
import collections
import datetime as dt
import itertools
import json
import time
from pathlib import Path

import numba as nb
import numpy as np

GEM = {
    "א":1,"ב":2,"ג":3,"ד":4,"ה":5,"ו":6,"ז":7,"ח":8,"ט":9,
    "י":10,"כ":20,"ך":20,"ל":30,"מ":40,"ם":40,"נ":50,"ן":50,"ס":60,
    "ע":70,"פ":80,"ף":80,"צ":90,"ץ":90,"ק":100,"ר":200,"ש":300,"ת":400,
}
DIGIT_WORD = {0:"אפס",1:"אחד",2:"שתיים",3:"שלוש",4:"ארבע",5:"חמש",6:"שש",7:"שבע",8:"שמונה",9:"תשע"}
UNITS = {0:"",1:"אחד",2:"שתיים",3:"שלוש",4:"ארבע",5:"חמש",6:"שש",7:"שבע",8:"שמונה",9:"תשע"}
TEENS = {10:"עשר",11:"אחת עשרה",12:"שתים עשרה",13:"שלוש עשרה",14:"ארבע עשרה",15:"חמש עשרה",16:"שש עשרה",17:"שבע עשרה",18:"שמונה עשרה",19:"תשע עשרה"}
TENS = {20:"עשרים",30:"שלושים",40:"ארבעים",50:"חמישים",60:"שישים",70:"שבעים",80:"שמונים",90:"תשעים"}
HUNDREDS = {100:"מאה",200:"מאתיים",300:"שלוש מאות",400:"ארבע מאות",500:"חמש מאות",600:"שש מאות",700:"שבע מאות",800:"שמונה מאות",900:"תשע מאות"}
THOUSANDS = {1000:"אלף",2000:"אלפיים",3000:"שלושת אלפים",4000:"ארבעת אלפים",5000:"חמשת אלפים",6000:"ששת אלפים",7000:"שבעת אלפים",8000:"שמונת אלפים",9000:"תשעת אלפים"}
MONTHS = {1:"בינואר",2:"בפברואר",3:"במרץ",4:"באפריל",5:"במאי",6:"ביוני",7:"ביולי",8:"באוגוסט",9:"בספטמבר",10:"באוקטובר",11:"בנובמבר",12:"בדצמבר"}
YEAR_WORDS = "אלפיים עשרים ושש"
YEAR = 2026


def g(text: str) -> int:
    return sum(GEM.get(ch, 0) for ch in text)


def under_100(n: int) -> str:
    if n == 0: return ""
    if n < 10: return UNITS[n]
    if n < 20: return TEENS[n]
    tens = (n // 10) * 10
    unit = n % 10
    return TENS[tens] + ((" ו" + UNITS[unit]) if unit else "")


def number_words(n: int) -> str:
    if not 0 <= n <= 9999:
        raise ValueError("number_words supports 0..9999")
    if n == 0: return "אפס"
    parts: list[str] = []
    thousands = (n // 1000) * 1000
    remainder = n % 1000
    if thousands: parts.append(THOUSANDS[thousands])
    hundreds = (remainder // 100) * 100
    remainder %= 100
    if hundreds: parts.append(HUNDREDS[hundreds])
    if remainder:
        words = under_100(remainder)
        if remainder < 10 and (thousands or hundreds): words = "ו" + words
        parts.append(words)
    return " ".join(parts)


def date_value(date: dt.date) -> int:
    return g(f"{under_100(date.day)} {MONTHS[date.month]} {YEAR_WORDS}")


def reverse_int(n: int) -> int:
    return int(str(int(n))[::-1])


# Shared lookup tables.
MAX_N = 9999
WHOLE = np.zeros(MAX_N + 1, dtype=np.int32)
DIGITS = np.zeros(MAX_N + 1, dtype=np.int32)
REVERSE = np.zeros(MAX_N + 1, dtype=np.int32)
for _n in range(MAX_N + 1):
    WHOLE[_n] = g(number_words(_n))
    DIGITS[_n] = sum(g(DIGIT_WORD[int(ch)]) for ch in str(_n))
    REVERSE[_n] = reverse_int(_n)

DATES = [dt.date(YEAR, 1, 1) + dt.timedelta(days=i) for i in range(365)]
DATE_FULL = np.array([date_value(d) for d in DATES], dtype=np.int32)
DATE_COMPACT = np.array([int(f"{d.day}{d.month}") for d in DATES], dtype=np.int32)
DATE_PADDED = np.array([int(f"{d.day}{d.month:02d}") for d in DATES], dtype=np.int32)
DATE_NUM = np.array([d.day + d.month + d.year for d in DATES], dtype=np.int32)

# Plausible hotel rooms: floors 1..25 and room suffixes 01..50.
FLOORS=[]; SUFFIXES=[]; ROOMS=[]; ROOM_WORD=[]; FLOOR_WORD=[]
ROOM_DESC=[]; ROOM_FLOOR_PHRASE=[]; ROOM_PHRASE=[]; SPLIT=[]
for _floor in range(1, 26):
    for _suffix in range(1, 51):
        _room = _floor * 100 + _suffix
        _rw = g(number_words(_room)); _fw = g(number_words(_floor))
        FLOORS.append(_floor); SUFFIXES.append(_suffix); ROOMS.append(_room)
        ROOM_WORD.append(_rw); FLOOR_WORD.append(_fw)
        ROOM_DESC.append(g("מס חדר") + _rw)
        ROOM_FLOOR_PHRASE.append(g("מס חדר") + _rw + g("קומה") + _fw)
        ROOM_PHRASE.append(g("חדר") + _rw)
        SPLIT.append(_suffix * 100 + _floor)
FLOORS=np.array(FLOORS,dtype=np.int32); SUFFIXES=np.array(SUFFIXES,dtype=np.int32)
ROOMS=np.array(ROOMS,dtype=np.int32); ROOM_WORD=np.array(ROOM_WORD,dtype=np.int32)
FLOOR_WORD=np.array(FLOOR_WORD,dtype=np.int32); ROOM_DESC=np.array(ROOM_DESC,dtype=np.int32)
ROOM_FLOOR_PHRASE=np.array(ROOM_FLOOR_PHRASE,dtype=np.int32)
ROOM_PHRASE=np.array(ROOM_PHRASE,dtype=np.int32); SPLIT=np.array(SPLIT,dtype=np.int32)

AGE_WORD=np.zeros(91,dtype=np.int32)
for _a in range(18,91): AGE_WORD[_a]=g(number_words(_a))
DIGIT_VALUES=np.array([g(DIGIT_WORD[i]) for i in range(10)],dtype=np.int32)
PREFIX_VALUE=sum(g(DIGIT_WORD[int(ch)]) for ch in "9723")


def observed_conditions() -> list[bool]:
    H=435; surname=292; first_a=251; first_e=320
    A=first_a+surname; E=first_e+surname; age_a=76; birth=1949; age_e=44
    i1=(dt.date(2026,7,26)-DATES[0]).days; i2=i1+4; i3=i1+5
    ridx=int(np.where((FLOORS==12)&(SUFFIXES==6))[0][0])
    room=int(ROOMS[ridx]); floor=int(FLOORS[ridx])
    B=g("חדר מספר")+int(ROOM_WORD[ridx])+H+g("תל אביב קומה")+int(FLOOR_WORD[ridx])
    Q1=g("מהו מספר חדר ששהה")+A+g("במלון")+H
    Q2=Q1+g("תל אביב")
    age_phrase=E+g("בן")+g(number_words(age_e)); node=Q2+A
    route=int(DATE_PADDED[i1])-int(ROOM_DESC[ridx])-A
    short_diff=int(WHOLE[5450]-DIGITS[5450])
    long_value=sum(g(DIGIT_WORD[int(ch)]) for ch in "97235202552")
    return [
        B==int(DATE_FULL[i1])+room+floor+int(DATE_COMPACT[i1]),
        B-int(WHOLE[B])-int(DIGITS[B])==2*int(DATE_COMPACT[i1]),
        route==2*int(DATE_COMPACT[i1]) and reverse_int(route)==H,
        int(ROOM_FLOOR_PHRASE[ridx])-2*A-int(DATE_COMPACT[i1])-room-floor==H,
        YEAR-(B-int(DATE_PADDED[i1]))+age_a==A,
        B-birth-YEAR+age_a==int(DATE_COMPACT[i1]),
        B-Q1-A-58==room,
        B-int(DATE_NUM[i1])-int(DATE_NUM[i2])==age_e,
        int(DATE_FULL[i1])+int(DATE_FULL[i2])-B-E==age_e,
        Q2-age_phrase-floor-first_e==room,
        node==int(DATE_FULL[i2])+room,
        node-int(DIGITS[node])==int(ROOM_PHRASE[ridx]),
        room-floor-short_diff==int(DATE_COMPACT[i1]),
        reverse_int(long_value)==int(ROOM_FLOOR_PHRASE[ridx]),
        reverse_int(int(DATE_FULL[i3]))==int(SPLIT[ridx]),
        E==int(SPLIT[ridx]),
    ]


def full_event_monte_carlo(trials: int, seed: int=2026073102, batch: int=250_000):
    rng=np.random.default_rng(seed); counts=np.zeros(17,dtype=np.int64)
    max_score=0
    for start in range(0,trials,batch):
        n=min(batch,trials-start)
        i1=rng.integers(0,360,size=n); i2=i1+4; i3=i1+5
        ridx=rng.integers(0,len(ROOMS),size=n)
        H=rng.integers(100,1000,size=n)
        surname=rng.integers(50,401,size=n)
        first_a=rng.integers(20,501,size=n); first_e=rng.integers(20,501,size=n)
        A=first_a+surname; E=first_e+surname
        age_a=rng.integers(18,91,size=n); birth=YEAR-age_a-1
        age_e=rng.integers(18,91,size=n)
        short=rng.integers(1000,10000,size=n)
        suffix=rng.integers(0,10,size=(n,7))
        floor=FLOORS[ridx]; room=ROOMS[ridx]
        B=g("חדר מספר")+ROOM_WORD[ridx]+H+g("תל אביב קומה")+FLOOR_WORD[ridx]
        Q1=g("מהו מספר חדר ששהה")+A+g("במלון")+H
        Q2=Q1+g("תל אביב"); node=Q2+A
        age_phrase=E+g("בן")+AGE_WORD[age_e]
        route=DATE_PADDED[i1]-ROOM_DESC[ridx]-A
        short_diff=WHOLE[short]-DIGITS[short]
        long_value=PREFIX_VALUE+DIGIT_VALUES[suffix].sum(axis=1)
        c=np.empty((16,n),dtype=np.bool_)
        c[0]=B==DATE_FULL[i1]+room+floor+DATE_COMPACT[i1]
        c[1]=B-WHOLE[B]-DIGITS[B]==2*DATE_COMPACT[i1]
        valid=(route>=0)&(route<=MAX_N); c[2]=False
        c[2,valid]=(route[valid]==2*DATE_COMPACT[i1][valid])&(REVERSE[route[valid]]==H[valid])
        c[3]=ROOM_FLOOR_PHRASE[ridx]-2*A-DATE_COMPACT[i1]-room-floor==H
        c[4]=YEAR-(B-DATE_PADDED[i1])+age_a==A
        c[5]=B-birth-YEAR+age_a==DATE_COMPACT[i1]
        c[6]=B-Q1-A-58==room
        c[7]=B-DATE_NUM[i1]-DATE_NUM[i2]==age_e
        c[8]=DATE_FULL[i1]+DATE_FULL[i2]-B-E==age_e
        c[9]=Q2-age_phrase-floor-first_e==room
        c[10]=node==DATE_FULL[i2]+room
        valid=(node>=0)&(node<=MAX_N); c[11]=False
        c[11,valid]=node[valid]-DIGITS[node[valid]]==ROOM_PHRASE[ridx][valid]
        c[12]=room-floor-short_diff==DATE_COMPACT[i1]
        valid=(long_value>=0)&(long_value<=MAX_N); c[13]=False
        c[13,valid]=REVERSE[long_value[valid]]==ROOM_FLOOR_PHRASE[ridx][valid]
        c[14]=REVERSE[DATE_FULL[i3]]==SPLIT[ridx]
        c[15]=E==SPLIT[ridx]
        score=c.sum(axis=0)
        counts += np.bincount(score,minlength=17)
        max_score=max(max_score,int(score.max()))
    return counts,max_score


def exact_room_date_grid():
    H=435; A=543; E=612; first_e=320; age_a=76; birth=1949; age_e=44
    Q1=g("מהו מספר חדר ששהה")+A+g("במלון")+H
    Q2=Q1+g("תל אביב"); node=Q2+A
    age_phrase=E+g("בן")+AGE_WORD[age_e]
    short_diff=int(WHOLE[5450]-DIGITS[5450])
    long_value=sum(g(DIGIT_WORD[int(ch)]) for ch in "97235202552")
    hist=collections.Counter(); winners=[]
    for i1 in range(360):
        i2=i1+4; i3=i1+5; floor=FLOORS; room=ROOMS
        B=g("חדר מספר")+ROOM_WORD+H+g("תל אביב קומה")+FLOOR_WORD
        route=DATE_PADDED[i1]-ROOM_DESC-A
        c=np.empty((16,len(ROOMS)),dtype=np.bool_)
        c[0]=B==DATE_FULL[i1]+room+floor+DATE_COMPACT[i1]
        c[1]=B-WHOLE[B]-DIGITS[B]==2*DATE_COMPACT[i1]
        valid=(route>=0)&(route<=MAX_N); c[2]=False
        c[2,valid]=(route[valid]==2*DATE_COMPACT[i1])&(REVERSE[route[valid]]==H)
        c[3]=ROOM_FLOOR_PHRASE-2*A-DATE_COMPACT[i1]-room-floor==H
        c[4]=YEAR-(B-DATE_PADDED[i1])+age_a==A
        c[5]=B-birth-YEAR+age_a==DATE_COMPACT[i1]
        c[6]=B-Q1-A-58==room
        c[7]=B-DATE_NUM[i1]-DATE_NUM[i2]==age_e
        c[8]=DATE_FULL[i1]+DATE_FULL[i2]-B-E==age_e
        c[9]=Q2-age_phrase-floor-first_e==room
        c[10]=node==DATE_FULL[i2]+room
        c[11]=node-DIGITS[node]==ROOM_PHRASE
        c[12]=room-floor-short_diff==DATE_COMPACT[i1]
        c[13]=REVERSE[long_value]==ROOM_FLOOR_PHRASE
        c[14]=REVERSE[DATE_FULL[i3]]==SPLIT
        c[15]=E==SPLIT
        scores=c.sum(axis=0)
        for s,count in enumerate(np.bincount(scores,minlength=17)):
            if count: hist[s]+=int(count)
        for j in np.where(scores==16)[0]:
            winners.append({"date":DATES[i1].isoformat(),"floor":int(FLOORS[j]),"suffix":int(SUFFIXES[j]),"room":int(ROOMS[j])})
    return hist,winners


def score_assignment(H,A,E,first_e,date_tuple,age_a,age_e):
    i1=(date_tuple[0]-DATES[0]).days; i2=(date_tuple[1]-DATES[0]).days; i3=(date_tuple[2]-DATES[0]).days
    ridx=int(np.where((FLOORS==12)&(SUFFIXES==6))[0][0]); room=int(ROOMS[ridx]); floor=int(FLOORS[ridx])
    birth=YEAR-age_a-1
    B=g("חדר מספר")+int(ROOM_WORD[ridx])+H+g("תל אביב קומה")+int(FLOOR_WORD[ridx])
    Q1=g("מהו מספר חדר ששהה")+A+g("במלון")+H; Q2=Q1+g("תל אביב")
    age_phrase=E+g("בן")+g(number_words(age_e)); node=Q2+A
    route=int(DATE_PADDED[i1])-int(ROOM_DESC[ridx])-A
    short_diff=int(WHOLE[5450]-DIGITS[5450])
    long_value=sum(g(DIGIT_WORD[int(ch)]) for ch in "97235202552")
    c=[
        B==int(DATE_FULL[i1])+room+floor+int(DATE_COMPACT[i1]),
        B-int(WHOLE[B])-int(DIGITS[B])==2*int(DATE_COMPACT[i1]),
        route>=0 and route<=MAX_N and route==2*int(DATE_COMPACT[i1]) and int(REVERSE[route])==H,
        int(ROOM_FLOOR_PHRASE[ridx])-2*A-int(DATE_COMPACT[i1])-room-floor==H,
        YEAR-(B-int(DATE_PADDED[i1]))+age_a==A,
        B-birth-YEAR+age_a==int(DATE_COMPACT[i1]),
        B-Q1-A-58==room,
        B-int(DATE_NUM[i1])-int(DATE_NUM[i2])==age_e,
        int(DATE_FULL[i1])+int(DATE_FULL[i2])-B-E==age_e,
        Q2-age_phrase-floor-first_e==room,
        node==int(DATE_FULL[i2])+room,
        node-int(DIGITS[node])==int(ROOM_PHRASE[ridx]),
        room-floor-short_diff==int(DATE_COMPACT[i1]),
        int(REVERSE[long_value])==int(ROOM_FLOOR_PHRASE[ridx]),
        int(REVERSE[int(DATE_FULL[i3])])==int(SPLIT[ridx]),
        E==int(SPLIT[ridx]),
    ]
    return sum(c)


def exact_role_permutation():
    names=[435,543,612,320]
    dates=[dt.date(2026,7,26),dt.date(2026,7,30),dt.date(2026,7,31)]
    ages=[76,44]
    hist=collections.Counter(); winners=[]
    for nv in itertools.permutations(names):
        for dv in itertools.permutations(dates):
            for av in itertools.permutations(ages):
                score=score_assignment(*nv,dv,*av)
                hist[score]+=1
                if score==16:
                    winners.append({"names":nv,"dates":[d.isoformat() for d in dv],"ages":av})
    return hist,winners


@nb.njit
def density_metrics(arr, whole, digits, reverse):
    n=arr.shape[0]
    r12=r13=r22=rrev=rdbl=rt1=rt2=0
    for i in range(n):
        for j in range(i+1,n):
            s=arr[i]+arr[j]
            for k in range(n):
                if k!=i and k!=j and arr[k]==s: r12+=1
    for i in range(n):
        for j in range(i+1,n):
            for k in range(j+1,n):
                s=arr[i]+arr[j]+arr[k]
                for l in range(n):
                    if l!=i and l!=j and l!=k and arr[l]==s: r13+=1
    for i in range(n):
        for j in range(i+1,n):
            s=arr[i]+arr[j]
            for k in range(i+1,n):
                for l in range(k+1,n):
                    if k==i or k==j or l==i or l==j: continue
                    if arr[k]+arr[l]==s: r22+=1
    for i in range(n):
        for j in range(i+1,n):
            if reverse[arr[i]]==arr[j] or reverse[arr[j]]==arr[i]: rrev+=1
            if arr[i]==2*arr[j] or arr[j]==2*arr[i]: rdbl+=1
    for i in range(n):
        x=arr[i]
        if 1000<=x<whole.shape[0]:
            t1=x-digits[x]; t2=x-whole[x]-digits[x]
            for j in range(n):
                if i!=j:
                    if arr[j]==t1: rt1+=1
                    if arr[j]==t2: rt2+=1
    return r12+r13+r22+rrev+rdbl+rt1+rt2


@nb.njit(parallel=True)
def batch_density(arrs, whole, digits, reverse):
    out=np.empty(arrs.shape[0],dtype=np.int32)
    for i in nb.prange(arrs.shape[0]): out[i]=density_metrics(arrs[i],whole,digits,reverse)
    return out


def free_relation_density(generated: int, seed: int=1357911, batch: int=50_000):
    observed=np.array([4166,2681,1206,12,267,2607,1530,543,534,435,3006,1949,2026,76,2359,58,2804,1266,320,2146,1486,2059,2063,44,2141,612,3347,1917,1430,2037,1110,927,6003,2160],dtype=np.int32)
    observed_score=int(density_metrics(observed,WHOLE[:7000],DIGITS[:7000],REVERSE[:7000]))
    rng=np.random.default_rng(seed); hist=collections.Counter(); accepted=0
    for start in range(0,generated,batch):
        m=min(batch,generated-start)
        arr=np.empty((m,34),dtype=np.int32)
        arr[:,:4]=rng.integers(10,100,size=(m,4))
        arr[:,4:11]=rng.integers(100,1000,size=(m,7))
        arr[:,11:]=rng.integers(1000,7000,size=(m,23))
        unique=(np.all(np.diff(np.sort(arr[:,:4],axis=1),axis=1)!=0,axis=1)&
                np.all(np.diff(np.sort(arr[:,4:11],axis=1),axis=1)!=0,axis=1)&
                np.all(np.diff(np.sort(arr[:,11:],axis=1),axis=1)!=0,axis=1))
        arr=arr[unique]; accepted+=len(arr)
        if len(arr)==0: continue
        scores=batch_density(arr,WHOLE[:7000],DIGITS[:7000],REVERSE[:7000])
        for s,count in enumerate(np.bincount(scores)):
            if count: hist[s]+=int(count)
    tail=sum(c for s,c in hist.items() if s>=observed_score)
    return observed_score,accepted,hist,tail


def chronological_holdout():
    d1=dt.date(2026,7,26); i1=(d1-DATES[0]).days
    passes=[]
    for i2,d2 in enumerate(DATES[:-1]):
        i3=i2+1
        date30=(4166-int(DATE_NUM[i1])-int(DATE_NUM[i2])==44 and
                int(DATE_FULL[i1])+int(DATE_FULL[i2])-4166-612==44 and
                2804+543==int(DATE_FULL[i2])+1206)
        next_day=(int(REVERSE[int(DATE_FULL[i3])])==612)
        if date30 and next_day: passes.append((d2.isoformat(),DATES[i3].isoformat()))
    short_hits=sum(1 for n in range(1000,10000) if int(WHOLE[n]-DIGITS[n])==927)
    # Exact DP for the seven digits following prefix 9723.
    dist=collections.Counter({PREFIX_VALUE:1})
    for _ in range(7):
        nxt=collections.Counter()
        for total,count in dist.items():
            for dv in DIGIT_VALUES: nxt[total+int(dv)]+=count
        dist=nxt
    long_hits=dist[6003]
    joint=(len(passes)/364)*(short_hits/9000)*(long_hits/10_000_000)
    return {"date_sequences":passes,"short_hits":short_hits,"long_hits":long_hits,"joint_uniform_probability":joint}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--full-trials",type=int,default=1_000_000)
    p.add_argument("--density-generated",type=int,default=100_000)
    p.add_argument("--json-output",default="")
    args=p.parse_args()
    started=time.time()
    observed=observed_conditions()
    full_hist,full_max=full_event_monte_carlo(args.full_trials)
    grid_hist,grid_winners=exact_room_date_grid()
    perm_hist,perm_winners=exact_role_permutation()
    density_score,density_accepted,density_hist,density_tail=free_relation_density(args.density_generated)
    holdout=chronological_holdout()
    result={
        "observed_score":sum(observed),"observed_conditions":observed,
        "full_event":{"trials":args.full_trials,"histogram":{str(i):int(c) for i,c in enumerate(full_hist) if c},"maximum":full_max,"zero_hit_95pct_upper_bound":3/args.full_trials},
        "room_date_grid":{"combinations":450000,"histogram":dict(sorted(grid_hist.items())),"winners":grid_winners},
        "role_permutation":{"permutations":288,"histogram":dict(sorted(perm_hist.items())),"winners":perm_winners},
        "free_relation_density":{"observed_score":density_score,"generated":args.density_generated,"accepted_unique":density_accepted,"tail_count":density_tail,"tail_rate":density_tail/density_accepted,"maximum":max(density_hist),"histogram":dict(sorted(density_hist.items()))},
        "chronological_holdout":holdout,
        "runtime_seconds":time.time()-started,
    }
    text=json.dumps(result,ensure_ascii=False,indent=2)
    print(text)
    if args.json_output: Path(args.json_output).write_text(text,encoding="utf-8")

if __name__=="__main__": main()
