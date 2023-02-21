import pandas as pd
import matplotlib.pyplot as plt

model = 'davinci2'
df2 = pd.DataFrame(pd.read_csv('user-study/'+model+'.csv'), columns=["sample_id","c",  "n", "e"])
c = ['lightcoral',  'royalblue', 'mediumseagreen']


for _, row in df2.iterrows():
    idx = row['sample_id']
    print('hi')
    df = pd.DataFrame({'c': [], 'n': [], 'e':[]})
    df.at[idx, 'c'] = row['c']
    df.at[idx, 'n'] = row['n']
    df.at[idx, 'e'] = row['e']

    ax = df.plot.barh(stacked=True, width=0.1, color=c);
    plt.legend('',frameon=False)

    #plt.show()
    plt.savefig('img.png')


    import cv2
    img = cv2.imread("img.png")
    crop_img = img[225:270, 80:552]
    #cv2.imshow("cropped", crop_img)
    cv2.waitKey(0)
    isWritten = cv2.imwrite('user-study/'+model+'/'+str(int(idx))+'.png', crop_img)