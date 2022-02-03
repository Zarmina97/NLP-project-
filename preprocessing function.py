def emojis_replacement(col):
  for index, i in enumerate(col):
    i=str(i).replace("�", "")
    for e, m in zip(df_emo.emoji, df_emo.meaning):
      if e in i:
        i=i.replace(e, m+' ')
        i=i.replace('_', '')
    col.loc[index]=i
    
    
def lower_column(column):
  for col in [column]:
    df[col]=df[col].str.lower()