#!/usr/bin/env python
# coding: utf-8

# In[1]:


#imports
import re


# In[3]:


# for CK2BN list of BN that will never be formed in bangla
# ড়ঢ়শষঋ ৎ 'ৃ' 
NUM_CK_BN_MAP = {
'𑄶':'০',
'𑄷':'১',
'𑄸':'২',
'𑄹':'৩',
'𑄺':'৪',
'𑄻':'৫',
'𑄼':'৬',
'𑄽':'৭',
'𑄾':'৮',
'𑄿':'৯'
}

CONS_CK_BN_MAP = {
'𑄇':'ক',
'𑄈':'খ',
'𑄉':'গ',
'𑄊':'ঘ',
'𑄋':'ঙ',
'𑄌':'চ',
'𑄍':'ছ',
'𑄎':'জ',
'𑄏':'ঝ',
'𑄐':'ঞ',
'𑄑':'ট',
'𑄒':'ঠ',
'𑄓':'ড',
'𑄔':'ঢ',
'𑄕':'ণ',
'𑄖':'ত',
'𑄗':'থ',
'𑄘':'দ',
'𑄙':'ধ',
'𑄚':'ন',
'𑄛':'প',
'𑄜':'ফ',
'𑄝':'ব',
'𑄞':'ভ',
'𑄟':'ম',
'𑄡':'য',
'𑄢':'র',
'𑄣':'ল',
'𑄥':'স',
'𑄦':'হ',
'𑄠':'য়',
'𑅄':'ল',
'𑄃':'অ' # although it is actual & only vowel, it acts like consonants i.e. can have other vowel signs attached
}

VOWELS_CK_BN_MAP = {
'𑄄':'ই',
'𑄅':'উ',
'𑄆':'এ',
'𑄤':'ওআ' # can't have attached anything
}

SIGNS_CK_BN_MAP = {
'𑄧':'া',
'𑄨':'ি',
'𑄩':'ী',
'𑄪':'ু',
'𑄫':'ূ',
'𑄬':'ে',
'𑄰':'ৈ',
'𑄮':'ো',
'𑄯':'ৌ',
'𑄴':'্',
'𑄳':'্'
}

OTHER_SIGNS_CK_BN_MAP = {
'𑄀':'ঁ',
'𑄁':'ং',
'𑄂':'ঃ',
'𑄭':'ই',
'𑄱':'োআ',
'𑄲':'োআ',
'𑅆':'ই',
'𑅅':'', # its just pronounced longer
'𑄳𑄆':None, # this is replaced in pre_process
 '𑄳𑄅':'উ' # handled in pre_process
}

CK_BN_MAP = {}
CK_BN_MAP.update(NUM_CK_BN_MAP)
CK_BN_MAP.update(CONS_CK_BN_MAP)
CK_BN_MAP.update(VOWELS_CK_BN_MAP)
CK_BN_MAP.update(SIGNS_CK_BN_MAP)
CK_BN_MAP.update(OTHER_SIGNS_CK_BN_MAP)

BN_AA_KAR = 'া'
CK_AA_KAR = '𑄧'
CK_VIRAMA = '𑄳'
CK_MAJYA = '𑄴'


# In[5]:


# BN2CK
NUM_BN_CK_MAP = {
'০':'𑄶',
'১':'𑄷',
'২':'𑄸',
'৩':'𑄹',
'৪':'𑄺',
'৫':'𑄻',
'৬':'𑄼',
'৭':'𑄽',
'৮':'𑄾',
'৯':'𑄿'  
}

CONS_BN_CK_MAP = {
'ক':'𑄇',
'খ':'𑄈',
'গ':'𑄉',
'ঘ':'𑄊',
'ঙ':'𑄋',
'চ':'𑄌',
'ছ':'𑄍',
'জ':'𑄎',
'ঝ':'𑄏',
'ঞ':'𑄐',
'ট':'𑄑',
'ঠ':'𑄒',
'ড':'𑄓',
'ঢ':'𑄔',
'ণ':'𑄕',
'ত':'𑄖',
'থ':'𑄗',
'দ':'𑄘',
'ধ':'𑄙',
'ন':'𑄚',
'প':'𑄛',
'ফ':'𑄜',
'ব':'𑄝',
'ভ':'𑄞',
'ম':'𑄟',
'য':'𑄡',
'র':'𑄢',
'ল':'𑄣',
'স':'𑄥',
'শ':'𑄥',
'ষ':'𑄥',
'হ':'𑄦',
'য়':'𑄠',
'ড়':'𑄢',
'ঢ়':'𑄢',
'অ':'𑄃' # can have other signs attached
}

VOWELS_BN_CK_MAP = {
'আ':'𑄃',
'ই':'𑄃𑄨',
'ঈ':'𑄃𑄩',
'উ':'𑄃𑄪',
'ঊ':'𑄃𑄫',
'ঋ':'𑄢𑄨',
'এ':'𑄃𑄬',
'ঐ':'𑄃𑄰',
'ও':'𑄃𑄮',
'ঔ':'𑄃𑄯',
'ৎ':'𑄖𑄴' # similar to vowels since it can't have other vowel signs associated with it 
}

SIGNS_BN_CK_MAP = {
'ি':'𑄨',
'ী':'𑄩',
'ু':'𑄪',
'ূ':'𑄫',
'ে':'𑄬',
'ৈ':'𑄰',
'ো':'𑄮',
'ৌ':'𑄯',
'া':'𑄧',
'্':'𑄴', # keeping here for similarity # this will be resolved actualy during post_process()
'ৃ':'𑄳𑄢𑄨' # no ঋ-কার in chakma -> so replacing it with ra-fhola + i-kar (no extra other kar can be attached with it) # ask expert if ok?
}

OTHER_SIGNS_BN_CK_MAP = {
'ঁ':'𑄀',
'ং':'𑄁',
'ঃ':'𑄂'
}

BN_CK_MAP = {}
BN_CK_MAP.update(NUM_BN_CK_MAP)
BN_CK_MAP.update(CONS_BN_CK_MAP)
BN_CK_MAP.update(VOWELS_BN_CK_MAP)
BN_CK_MAP.update(SIGNS_BN_CK_MAP)
BN_CK_MAP.update(OTHER_SIGNS_BN_CK_MAP)

#common in both
BN_AA_KAR = 'া'
CK_AA_KAR = '𑄧'
CK_VIRAMA = '𑄳'
CK_MAJYA = '𑄴'

# for our project only
exception_list_ka = set(['কম্পিউটার্',
'কোম্পানিবু',
'কোম্পানি',
'কিলোমিটার্',
'কোম্পানিবোত্তে',
'কন্সার্ট',
'কম্পানিত্',
'কফি',
'কোম্পানিট্',
'ক্লাব্',
'কার্ড',
'কন্ডিশন্',
'কিলোমিটা।',
'কৃতজ্ঞ',
'কানাডাট্',
'কৃতজ্ঞ।',
'করিমে',
'করিম্',
'কঠিন্।',
'কলিগ্।',
'ক্রেডিট',
'কল্',
'কলিগ্',
'কানাডা',
'কুম্পোনিত্'])

# for our project only
exception_list_ha = set(['হোটোলঅ',
'হোটেল্',
'হাস্পাতাল্',
'হোটেলত্',
'হোটেলো'])


# In[10]:


def bn2ck_pre_correction(text):
    # correcting starting word starting with 'ক'/'হ' by 'খ' unless its non-chakma word
    # text = re.sub(r'(?<!\S)'+'ক','খ',text) - this will not work / will be bit complex - thus doing in loop
    text = text.split()
    result = []
    for word in text:
        if len(word) >= 1 and word[0] == 'ক' and word not in exception_list_ka:
            word = 'খ' + word[1:]
        if len(word) >= 1 and word[0] == 'হ' and word not in exception_list_ha:
            word = 'খ' + word[1:]
        result.append(word)
    text = ' '.join(result)
    
    
    return text

def bn2ck_pre_process(text):
    # replacing shunno + ra
    text = re.sub('ব়','র', text) # ba + dot
    text = re.sub('ড়','র', text) # da + dot 
    text = re.sub('ড়','র', text) 
    text = re.sub('ঢ়','র', text) # dha + dot 
    text = re.sub('ঢ়','র', text)
    text = re.sub('য়','য়', text) # ja + dot
    
    # not replacing different sa's/others as they can be in jukto bornos
    
    # fixing special juktoborns , skipping jukto related to 'ঞ'
    text = re.sub('ক্ষ', 'ক্খ', text) # ask expert if ok?
    text = re.sub('জ্ঞ', 'গ্গ', text) # ask expert if ok?
    
    return text

def bn2ck_store_and_reset(cg_list, signs_list, previous_cg, signs_temp_list):
    if previous_cg[0] != None:
        cg_list.append(previous_cg[0])
        signs_list.append(signs_temp_list[0])

    previous_cg[0] = None
    signs_temp_list[0] = []
    
def bn2ck_extract_bn(text):
    cg_list, signs_list = [], []
    
    previous_cg = [None]
    signs_temp_list = [[]]
    
    for c in text:
        if c in SIGNS_BN_CK_MAP.keys() or c in OTHER_SIGNS_BN_CK_MAP.keys():
            signs_temp_list[0].append(c)
        else: # coregrapheme/punctuations/unknown - keeps the unknown unconverted
            bn2ck_store_and_reset(cg_list, signs_list, previous_cg, signs_temp_list)
            previous_cg[0] = c
    
    # if not stored at the end
    bn2ck_store_and_reset(cg_list, signs_list, previous_cg, signs_temp_list) 
    
    return cg_list, signs_list

def bn2ck_translate_signs(signs_list, cg):
    result_list = []
    
    has_aa_kar = False
    has_other_kar = False
    for sign in signs_list:
        if sign == BN_AA_KAR:
            has_aa_kar = True     
        elif sign in SIGNS_BN_CK_MAP.keys() and cg in CONS_BN_CK_MAP.keys():
            has_other_kar = True
            result_list.append(BN_CK_MAP[sign])
        elif sign in OTHER_SIGNS_BN_CK_MAP.keys():
            result_list.append(BN_CK_MAP[sign])
            
    if not has_aa_kar and not has_other_kar and cg in CONS_BN_CK_MAP.keys():
        result_list = [CK_AA_KAR] + result_list
        
    return result_list
    

def bn2ck_translate(bn_cg_list, bn_signs_list):
    ck_cg_list, ck_signs_list = [], []
    for cg, signs_list in zip(bn_cg_list, bn_signs_list):
        ck_cg_list.append(BN_CK_MAP[cg] if cg in BN_CK_MAP.keys() else cg)
        
        signs_list = bn2ck_translate_signs(signs_list, cg)
        ck_signs_list.append(signs_list)
    
    return ck_cg_list, ck_signs_list

def bn2ck_generate_text(ck_cg_list, ck_signs_list):
    text = []
    for cg, signs_list in zip(ck_cg_list, ck_signs_list):
        text.append(cg)
        for sign in signs_list:
            text.append(sign)
    
    return ''.join(text)

def bn2ck_resolve_ha(text):
    # fix 'la' - it can't have majya on it, and its application totally different than other letters
    text = re.sub(r'𑄦𑄴', '𑄳𑄦', text)
    
    result = []
    idx = 0
    while idx < len(text):
        if text[idx] == '𑄳' and (idx+1) < len(text) and text[idx+1] == '𑄦':
            temp = []
            while (len(result)>=0 and (result[-1] in SIGNS_BN_CK_MAP.values() or result[-1] in OTHER_SIGNS_BN_CK_MAP.values())):
                temp.append(result.pop())
                
            if len(result) >= 0 and result[-1] == '𑄃': # skipping '𑄃' (no needed)
                result.pop()
                result.append('𑄦')
            else:
                result.append('𑄳')
                result.append('𑄦')
            
            while(len(temp)>0):
                result.append(temp.pop())
            idx+=2
        else:
            result.append(text[idx])
            idx+=1
    
    return ''.join(result)

def bn2ck_post_process(text):
    
    # fix fhola
    text = re.sub(r'𑄴𑄢', '𑄳𑄢', text)
    text = re.sub(r'𑄴𑄡', '𑄳𑄠', text)
    text = re.sub(r'𑄴𑄣', '𑄳𑄣', text)
    text = re.sub(r'𑄴𑄚', '𑄳𑄚', text)
    text = re.sub(r'𑄴𑄝', '𑄳𑄝', text)
    
    # 'ha' works differently in chakma
    text = bn2ck_resolve_ha(text)
    
    return text
    
def bn2ck(text):
    
    text = bn2ck_pre_correction(text) ## made only for our project.
    
    text = bn2ck_pre_process(text)
    
    bn_cg_list, bn_signs_list = bn2ck_extract_bn(text)

    ck_cg_list, ck_signs_list = bn2ck_translate(bn_cg_list, bn_signs_list)

    text = bn2ck_generate_text(ck_cg_list, ck_signs_list)
    
    text = bn2ck_post_process(text)
    
    return text


def bn2ck_list(texts):
    results = []

    for text in texts:
        results.append(bn2ck(text))

    return results


# In[15]:


def ck2bn_store_and_reset(cg_list, signs_list, previous_cg, signs_temp_list):
    if previous_cg[0] != None:
        cg_list.append(previous_cg[0])
        signs_list.append(signs_temp_list[0])

    previous_cg[0] = None
    signs_temp_list[0] = []

def ck2bn_extract_ck(text):
    cg_list, signs_list = [], []
    
    previous_cg = [None]
    signs_temp_list = [[]]
    
    for c in text:
        if c in SIGNS_CK_BN_MAP.keys() or c in OTHER_SIGNS_CK_BN_MAP.keys():
            signs_temp_list[0].append(c)
        else: # coregrapheme/punctuations/unknown - keeps the unknown unconverted
            ck2bn_store_and_reset(cg_list, signs_list, previous_cg, signs_temp_list)
            previous_cg[0] = c
    
    # if not stored at the end
    ck2bn_store_and_reset(cg_list, signs_list, previous_cg, signs_temp_list) 
    
    return cg_list, signs_list

def ck2bn_translate_signs(signs_list, cg):
    result_list = []
    has_aa_kar = False
    has_other_kar = False
    special_kar = None
    
    for sign in signs_list:
        if sign == CK_AA_KAR:
            has_aa_kar = True     
        elif sign in SIGNS_CK_BN_MAP.keys() and cg in CONS_CK_BN_MAP.keys():
            has_other_kar = True
            result_list.append(CK_BN_MAP[sign])
        elif sign in OTHER_SIGNS_CK_BN_MAP.keys():
            special_kar = sign
            result_list.append(CK_BN_MAP[sign])
            
    if not has_aa_kar and not has_other_kar and cg in CONS_CK_BN_MAP.keys():
        if special_kar == '𑅆':
            result_list = ['ো'] + result_list
        elif special_kar == '𑄱' or special_kar == '𑄲':
            result_list = result_list
        else:
            result_list = [BN_AA_KAR] + result_list
        
    return result_list
    

def ck2bn_translate(ck_cg_list, ck_signs_list):
    bn_cg_list, bn_signs_list = [], []
    for cg, signs_list in zip(ck_cg_list, ck_signs_list):
        bn_cg_list.append(CK_BN_MAP[cg] if cg in CK_BN_MAP.keys() else cg)
        
        signs_list = ck2bn_translate_signs(signs_list, cg)
        bn_signs_list.append(signs_list)
    
    return bn_cg_list, bn_signs_list

def ck2bn_generate_text(bn_cg_list, bn_signs_list):
    text = []
    for cg, signs_list in zip(bn_cg_list, bn_signs_list):
        text.append(cg)
        for sign in signs_list:
            text.append(sign)
    
    return ''.join(text)

def ck2bn_pre_process(text):
    text = re.sub('𑄳𑄄', '𑄳𑄆',text) # since it is similar
    text = re.sub('𑄴𑄳', '𑄴',text)
    text = re.sub('𑄬+', '𑄬',text)

    # fixing special kars
    idx = 0
    result = []
    while idx < len(text):
        if text[idx] == '𑄳' and (idx+1) < len(text)  and (text[idx+1] == '𑄅' or text[idx+1] == '𑄆' or text[idx+1] == '𑄦'):
            pos = idx+2
            while pos<len(text) and text[pos] in SIGNS_CK_BN_MAP.keys():
                result.append(text[pos])
                pos+=1
            
            if (text[idx+1] == '𑄅'):
                result.append('𑄅')
            elif (text[idx+1] == '𑄦'):
                result.append('𑄦'+'𑄳')
            else:
                result.append('𑅆')
                
            idx = pos
            continue
        
        result.append(text[idx])
        idx+=1

    # fixing single letter jukto borno
    text = ''.join(result)
    idx = 0
    result = []
    while idx < len(text):
        
        if idx > 0 and text[idx-1] in CONS_CK_BN_MAP.keys() and text[idx] == '𑄴' and (idx+1) < len(text)  and (text[idx+1] in SIGNS_CK_BN_MAP.keys()):
            result.append(text[idx])
            text = text[:idx + 1] + text[idx - 1] + text[idx + 1:]

            idx += 1
            continue

        result.append(text[idx])
        idx+=1
    return ''.join(result)
    
def ck2bn_post_process(text):
    # correct the vowels attached with AA
    text = re.sub('অা','আ', text)
    text = re.sub('অি','ই', text)
    text = re.sub('অী','ঈ', text) 
    text = re.sub('অু','উ', text) 
    text = re.sub('অূ','ঊ', text) 
    text = re.sub('অে','এ', text) 
    text = re.sub('অৈ','ঐ', text) 
    text = re.sub('অো','ও', text) 
    text = re.sub('অৌ','ঔ', text) 
    
    return text
    
def ck2bn(text):
    
    text = ck2bn_pre_process(text)
    
    ck_cg_list, ck_signs_list = ck2bn_extract_ck(text)
    
    bn_cg_list, bn_signs_list = ck2bn_translate(ck_cg_list, ck_signs_list)
    
    text = ck2bn_generate_text(bn_cg_list, bn_signs_list)
    
    text = ck2bn_post_process(text)
    
    return text

def ck2bn_list(texts):
    results = []

    for text in texts:
        results.append(ck2bn(text))

    return results
