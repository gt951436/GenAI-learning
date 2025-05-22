from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name:str = "Garv"    #default value in pydantic 
    age:Optional[int] = None
    email:EmailStr
    cgpa: float = Field(gt=0,lt=10,default = 5,description="Decimal value representing cgpa")
    
new_student = {'age':21,
               'email':'abc@gmail.com',
                'cgpa':9.19   # validation error --> bcoz cgpa>10
            } 
std = Student(**new_student)  # pydantic object

std_dict = dict(std)  # explicit conversion to dict object from pydantic object
print(std_dict)

std_json = std.model_dump_json()  # pydantic object to json object
print(std_json)