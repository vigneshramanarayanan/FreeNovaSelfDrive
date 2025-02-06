from Ultrasonic import *
ultrasonic=Ultrasonic() 
from Motor import *            
PWM=Motor()    
from servo import *
pwm=Servo()

def KeepRunningTillObstacle():
	data=ultrasonic.get_distance()   #Get the valued
	pwm.setServoPwm('0',90) #Set Servo 0 to look Straight
	pwm.setServoPwm('1',30) # Set Servo 1 to look slightly down to face the road 
	PWM.setMotorModel(0,0,0,0)	# Turn Off Motor First
	print(data)
	while(data>50):	# if more than 30 cm
		data=ultrasonic.get_distance()   #Get the value
		print ("Obstacle distance is "+str(data)+"CM")
		motorSpeed = 1000
		PWM.setMotorModel(motorSpeed,motorSpeed,motorSpeed,motorSpeed)
	PWM.setMotorModel(0,0,0,0)
	time.sleep(1)
	PWM.setMotorModel(-600,-600,-600,-600) #Reverse Car
	time.sleep(1)
	PWM.setMotorModel(0,0,0,0)
	time.sleep(1)

def TurnLeft():
	pwm.setServoPwm('0',90) #Set Servo 0 to look Straight	
	PWM.setMotorModel(-600,-600,800,800)       #Turn Left 
	print ("The car is turning left") 
	time.sleep(2)
	PWM.setMotorModel(0,0,0,0)       #Stop Motor
	time.sleep(1)			
				
				
				
def TurnRight():
	pwm.setServoPwm('0',90) #Set Servo 0 to look Straight	
	PWM.setMotorModel(800,800,-600,-600)       #Turn Right 
	print ("The car is turning Right") 
	time.sleep(2)
	PWM.setMotorModel(0,0,0,0)       #Stop Motor
	time.sleep(1)			
			
				
	
def StartCar():
	try:
		print ('Program is starting ... ')	
		
		KeepRunningTillObstacle()				
		#Obsctacle met		
		
		pwm.setServoPwm('0',30)	 #See left if there is space
		time.sleep(1)
		dataLeft=ultrasonic.get_distance()   #Get the Object in front value
		print ("Obstacle Left distance is "+str(dataLeft)+"CM")
		pwm.setServoPwm('0',150)	
		time.sleep(1)
		dataRight=ultrasonic.get_distance()   #Get the Object in front value
		print ("Obstacle Right distance is "+str(dataRight)+"CM")
		if(dataLeft > dataRight):
			pwm.setServoPwm('0',90) #Set Servo 0 to look Straight	
			data=ultrasonic.get_distance()   #Get the value
			print ("Obstacle distance is "+str(data)+"CM")			
			TurnLeft()
			time.sleep(1)
			data=ultrasonic.get_distance()   #Get the Object in front value
			print ("Obstacle distance is "+str(data)+"CM")
		else:	
			pwm.setServoPwm('0',90) #Set Servo 0 to look Straight	
			data=ultrasonic.get_distance()   #Get the value
			print ("Obstacle distance is "+str(data)+"CM")
			TurnRight()	
			time.sleep(1)
			data=ultrasonic.get_distance()   #Get the Object in front value
		
		StartCar()
		print ("\nEnd of program")
	except:		
		PWM.setMotorModel(0,0,0,0)
		print ("\nEnd of program")
		
if __name__ == '__main__':	
	StartCar()
