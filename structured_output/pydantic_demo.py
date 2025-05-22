from pydantic import BaseModel
from typing import Optional

class Student(BaseModel):
    name:str  ='Garv'  #default value in pydantic 
    age:Optional[int]= None
    
new_student = {'age':12} 
std = Student(**new_student)

print(std)
print(type(std))