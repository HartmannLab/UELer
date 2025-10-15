# viewer/observable.py

class Observable:
    """
    A class that represents an observable object, or "data" in the model-view paradigm.
    It can be observed by other objects.

    Attributes:
        _value: The initial value of the observable object.
        _observers: A list of observer functions that will be notified when the value changes.

    Methods:
        add_observer(observer):
            Adds an observer function to the list of observers.
        
        remove_observer(observer):
            Removes an observer function from the list of observers.
        
        notify_observers():
            Notifies all observer functions about the current value.
        
        value:
            Property that gets the current value of the observable object.
        
        value(new_value):
            Property setter that sets a new value for the observable object and notifies all observers.
    """
    def __init__(self, initial_value=None):
        self._value = initial_value
        self._observers = []

    def add_observer(self, observer):
        self._observers.append(observer)

    def remove_observer(self, observer):
        self._observers.remove(observer)

    def notify_observers(self):
        for observer in self._observers:
            observer(self._value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value
        self.notify_observers()