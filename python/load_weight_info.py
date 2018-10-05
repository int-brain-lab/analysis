import datetime

def get_weight_records(subjects, ai):
    """
    Determine whether the mouse has met the criteria for having learned
    
    Example:
        baseURL = 'https://alyx.internationalbrainlab.org/'
        ai = AlyxClient(username='miles', password=pwd, base_url=baseURL)
        records, info = get_weight_records(['ALK081', 'LEW010'], ai)
        
    Args: 
        subjects (list): List of subjects.
        ai (AlyxClient): An instance of the AlyxClient
        
    Returns:
        records (Data Frame): Data frame of weight and water records
        info (Data Frame): Data frame of subject information
        
    """
    s = ai.get('subjects?stock=False')
    rmKeys = ['actions_sessions','water_administrations','weighings','genotype']
    subject_info = []
    records = []
    weight_info = []
    for s in subjects:
        subj = ai.get('subjects/{}'.format(s))
        subject_info.append({key: subj[key] for key in subj if key not in rmKeys})
        endpoint = ('water-requirement/{}?start_date=2016-01-01&end_date={}'
                    .format(s, datetime.datetime.now().strftime('%Y-%m-%d')))
        wr = ai.get(endpoint)
        if wr['implant_weight']:
            iw = wr['implant_weight']
        else:
            iw = 0
        #TODO MultiIndex without None
        if not wr['records']:
            records.append(None)
        else:
            df = pd.DataFrame(wr['records'])
            df = (df.set_index(pd.DatetimeIndex(df['date']))
                  .drop('date', axis=1)
                  .assign(pct_weight = lambda x: 
                          (x['weight_measured']-iw) /
                          (x['weight_expected']-iw) 
                          if 'weight_measured' in x.columns.values 
                          else np.nan))
            records.append(df)
            wr.pop('records', None)
        weight_info.append(wr)

    
    info = (pd.DataFrame(weight_info)
            .merge(pd.DataFrame(subject_info), left_on='subject', right_on='nickname')
           .set_index('subject'))
    records = pd.concat(records, keys=subjects, names=['name', 'date'])
    return records, info
    return records