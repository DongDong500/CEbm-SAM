import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from email.header import Header



class _Email(object):

    def __init__(self, from_addr: list = ["singkuserver@gmail.com"], 
                to_addr: list = ["sdimivy014@korea.ac.kr"],
                subject: str = "Testing Mail system ... Do Not reply",
                msg: dict = {}, 
                attach: list = [],
                login_dir: str = '/home/dongik/src/login.json',
                ID = 'singkuserver', 
                ):
        """
        Args:
            from_addr: list of sender address
            to_addr: list of receiver address
            msg: Body message (type: dictionary)
            attach: list of attachment (images) directory
        """
        self.from_addr = from_addr
        self.to_addr = to_addr
        self.subject = subject
        self.message = msg
        self.attach = attach
        self.ID = ID
        self.login_dir = login_dir

        if os.path.exists(self.login_dir):
            with open(self.login_dir, "r") as f:
                self.users = json.load(f)
                self.PW = self.users[self.ID]
        else:
            raise RuntimeError("login info not exists:", self.login_dir)

    def send(self):
        """
        Args: Encryption Method
            TTL: smtplib.SMTP(smtp.gmail.com, 587)
            SSL: smtplib.SMTP_SSL(smtp.gmail.com, 465)
        """
        smtp = smtplib.SMTP('smtp.gmail.com', 587)
        smtp.starttls()
        smtp.login(self.ID, self.PW)

        msg = MIMEMultipart()
        msg['Subject'] = self.subject
        #msg.attach(MIMEText('Auto mail transfer system ...', 'plain'))
        
        if isinstance(self.message, dict):
            for key, val in self.message.items():
                if isinstance(val, dict):
                    if isinstance(key, int):
                        msg.attach(MIMEText(str(key) + '-th results', 'plain'))
                    elif isinstance(key, str):
                        msg.attach(MIMEText(key, 'plain'))
                    else:
                        raise NotImplementedError   
                    for skey, sval in val.items():
                        if not isinstance(sval, str):
                            msg.attach(MIMEText('\t' + skey + " : " + str(sval), 'plain'))
                        else:
                            msg.attach(MIMEText('\t' + skey + " : " + sval, 'plain'))
                elif isinstance(val, str):
                    msg.attach(MIMEText(val, 'plain'))
        elif isinstance(self.message, str):
            msg.attach(MIMEText(self.message, 'plain'))
        else:
            raise NotImplementedError

        smtp.sendmail(self.from_addr, self.to_addr, msg.as_string())

        smtp.quit()
        print("Email has been sent to '{}'".format(self.to_addr[-1]))

    def append_msg(self, msg):
        if isinstance(msg, list):
            self.message.append(msg)
        elif isinstance(msg, dict):
            self.message.update(msg)
        else:
            self.message.append(str(msg))

    def append_from_addr(self, addr):
        self.from_addr.append(addr)

    def append_to_addr(self, addr):
        self.to_addr.append(addr)

    def reset(self):
        self.message = []

class Email(object):

    def __init__(self, from_addr: list = ["singkuserver@gmail.com"], 
                to_addr: list = ["sdimivy014@korea.ac.kr"],
                subject: str = "Testing Mail system ... Do Not reply",
                msg: dict = {}, 
                attach: list = [],
                login_dir: str = '/home/dongik/src/login.json',
                ID = 'singkuserver', 
                ):
        """
        Args:
            from_addr: list of sender address
            to_addr: list of receiver address
            msg: Body message (type: dictionary)
            attach: list of attachment (images) directory
        """
        self.from_addr = from_addr
        self.to_addr = to_addr
        self.subject = subject
        self.message = msg
        self.attach = attach
        self.ID = ID
        self.login_dir = login_dir

        if os.path.exists(self.login_dir):
            with open(self.login_dir, "r") as f:
                self.users = json.load(f)
                self.PW = self.users[self.ID]
        else:
            raise RuntimeError("login info not exists:", self.login_dir)

    def send(self):
        """
        Args: Encryption Method
            TTL: smtplib.SMTP(smtp.gmail.com, 587)
            SSL: smtplib.SMTP_SSL(smtp.gmail.com, 465)
        """
        smtp = smtplib.SMTP('smtp.gmail.com', 587)
        smtp.starttls()
        smtp.login(self.ID, self.PW)

        msg = MIMEMultipart()
        msg.attach(MIMEText(self.message, 'html'))
        
        msg['Subject'] = self.subject
        
        

        smtp.sendmail(self.from_addr, self.to_addr, msg.as_string())

        smtp.quit()
        print("Email has been sent to '{}'".format(self.to_addr[-1]))



if __name__ == "__main__":
    import numpy as np , re
    import pandas as pd
    from tabulate import tabulate
    

    tb = [
        ['Jaccard', 'Precision', 'Recall', 'Specificity', 'Dice'],
        ['0.0000+0.0000', '0.0000+0.0000', '0.0000+0.0000', '0.0000+0.0000', '0.0000+0.0000']
    ]

    txt = """
    Table example
    {table}
    End
    """
    html = """
    <!DOCTYPE html>
    <html><body><p>Hello, Friend.</p>
    <p>Here is your data:</p>
    {table}
    <p>Regards,</p>
    <p>Me</p>
    </body></html>
    """

    txt = txt.format(table=tabulate(tb, headers='firstrow', tablefmt='grid'))
    html = html.format(table=tabulate(tb, headers='firstrow', tablefmt='html'))

    html_ = pd.DataFrame(
        np.random.normal(size = (1,5)),
        columns = [
            'Jaccard', 'Precision', 'Recall', 'Specificity', 'Dice'
        ], 
    ).to_html(index = False )
    
    #html_1 = pre + re.sub("\n", "",html_) + tail
    #html_1 = re.sub("\n", "", pre)


    sample_dict = {
        "Short Memo" : "str... str...",
        "F1 score" : {
            "background" : 0.98,
            "RoI" : 0.85
        },
        "html example" : html,
        "time elapsed" : "h:m:s.ms"
    }
    
    Email(from_addr=['singkuserver@gmail.com'],
            to_addr=['sdimivy014@korea.ac.kr'],
            msg=sample_dict,
            login_dir='/home/dongik/src/login.json',
            ID='singkuserver').send()
