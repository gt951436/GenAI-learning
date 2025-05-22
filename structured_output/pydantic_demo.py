from pydantic import BaseModel,EmailStr
from typing import Optional

class Student(BaseModel):
    name:str = "Garv"    #default value in pydantic 
    age:Optional[int] = None
    email:EmailStr
    
new_student = {'age':21,'email':'abc@gmail.com'} 
std = Student(**new_student)

print(std)