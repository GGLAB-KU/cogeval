import pandas as pd

acc_ids =[]
with open('/kuacc/users/mugekural/workfolder/dev/git/cogeval/user-study/prolific/useraccs_IDS.txt', 'r') as reader:
    for line in reader:
        line =line.strip()
        acc_ids.append(line)


df = pd.read_csv("results/SNLI-lalor/v3_fdf95_added_truefalse.csv")

for mid in [0,1,2,3,4,5]: 
    startid = 252
    numquests = 50
    qids = list(range(startid,startid+numquests))
    total_str = ''
    total_str += "[[AdvancedFormat]]" + "\n"
    total_str += "[[Block]]" + "\n"


    with open('results_model'+str(mid)+'.txt', 'w') as writer:
        for idx,row in df.iterrows():
            if idx >= len(qids):
                break
            print(idx)
            #for idx,row in df.iterrows():
            total_str += "[[Question:MC]]" + "\n"
            total_str +="[[ID:r"+str(idx+1)+"]]" + "\n"
            total_str +=  "\n"

            total_str += "[[AdvancedChoices]]" + "\n"
            total_str += "[[Choice]]" + "\n"
            total_str += "${q://QID"+str(qids[idx])+"/QuestionText}<br />" + "\n"
            total_str += "<br />" + "\n"
            total_str += "<strong>You &amp; Agent</strong><br />" + "\n"
            total_str += "<br />" + "\n"
            total_str += "${q://QID"+str(qids[idx])+"/ChoiceDescription/"+str(mid+1)+"}" + "\n"

            total_str += "[[Choice]]" + "\n"
            total_str += "${q://QID"+str(qids[idx])+"/QuestionText}<br />" + "\n"
            total_str += "<br />" + "\n"
            total_str += "<strong>You</strong>" + "\n"
            total_str += "<div id=\"useranswer"+str(idx+1)+"\">${q://QID"+str(qids[idx])+"/ChoiceGroup/SelectedChoices}</div>" + "\n"
            total_str += "<strong>Agent</strong>" + "\n"
            total_str +=  "\n"
            total_str += "<div id=\"agentanswer"+str(idx+1)+"\">${q://QID"+str(qids[idx])+"/ChoiceDescription/"+str(mid+1)+"}<br />" + "\n"
            total_str +="&nbsp;"
            total_str += "<div style=\"display:none\" id=\"useraccuracy"+str(idx+1)+"\">"+acc_ids[idx]+"</div>" + "\n"
            if len(row['true_machs']) == 2:
                total_str += "<div style=\"display:none\" id=\"modelaccuracy"+str(idx+1)+"\">0</div>" + "\n"
            elif (mid in  [int(mid) for mid in row['true_machs'][1:-1].split(',')]):
                total_str += "<div style=\"display:none\" id=\"modelaccuracy"+str(idx+1)+"\">1</div>" + "\n"
            else:
                total_str += "<div style=\"display:none\" id=\"modelaccuracy"+str(idx+1)+"\">0</div>" + "\n"
            total_str += "<br />" + "\n"
            total_str += "&nbsp;</div>" + "\n"
            total_str += "<br />" + "\n"
            total_str += "\n\n"
        writer.write(total_str)



