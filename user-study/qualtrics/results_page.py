import pandas as pd

df = pd.read_csv("user-study/subsets/subset1.csv")

for mid in [0,1,2,3,4,5]: 
    startid = 2
    numquests = 50
    qids = list(range(startid,startid+numquests))
    total_str = ''
    total_str += "[[AdvancedFormat]]" + "\n"
    total_str += "[[Block]]" + "\n"


    with open('results_subset1_model'+str(mid)+'.txt', 'w') as writer:
        for idx,row in df.iterrows():
            if idx >= len(qids):
                break
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
            total_str += "<div style=\"display:none\" id=\"useraccuracy"+str(idx+1)+"\">${q://QID"+str(qids[idx])+"/SelectedChoicesRecode}</div>" + "\n"
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



