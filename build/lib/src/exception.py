import sys # we need sys to get errors in runtime

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info() # execution info --> this will give 3 info in 3 variable im interested only in last variable so creating first 2 variables _ _
    # In exc_tb all the information like on which line error occured, in which line error occured everything what is error etc
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occured in python script name [{file_name}] line number [{line_number}] error message [{str(error)}]"

    return error_message

class customException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail)

    def __str__(self):
        return self.error_message
    

# if __name__=="__main__":
#     logging.info("Logging has started")

#     try:
#         a=1/0
#     except Exception as e:
#         logging.info('Dicision by zero') 
#         raise CustomException(e,sys)