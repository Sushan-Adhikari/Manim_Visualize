import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Import the generator functions
from manim_generator import generate_manim_code, save_code


# Dataset of functions organized by complexity (EXACTLY 537 total)
DATASET_FUNCTIONS = {
    "foundation": [
        # Basic polynomials and simple functions (130 samples)
        # Linear (20)
        ("x", "linear function"),
        ("2*x", "scaled linear"),
        ("3*x", "triple linear"),
        ("-x", "negative linear"),
        ("-2*x", "negative scaled linear"),
        ("x + 1", "linear with constant"),
        ("x + 3", "linear shifted up"),
        ("x - 2", "linear shifted down"),
        ("2*x + 1", "scaled linear with constant"),
        ("3*x - 4", "scaled linear shifted"),
        ("-x + 5", "negative linear shifted"),
        ("0.5*x", "half linear"),
        ("1.5*x", "one and half linear"),
        ("x/2", "linear divided"),
        ("x/3", "linear third"),
        ("-x/2", "negative half linear"),
        ("4*x", "quadruple linear"),
        ("5*x", "quintuple linear"),
        ("-3*x", "negative triple linear"),
        ("x + 10", "linear large shift"),
        
        # Quadratic (40)
        ("x^2", "basic quadratic"),
        ("2*x^2", "scaled quadratic"),
        ("3*x^2", "triple quadratic"),
        ("-x^2", "negative quadratic"),
        ("-2*x^2", "negative scaled quadratic"),
        ("0.5*x^2", "half quadratic"),
        ("x^2 + 1", "shifted quadratic"),
        ("x^2 + 2", "quadratic shifted up 2"),
        ("x^2 + 5", "quadratic shifted up 5"),
        ("x^2 - 1", "shifted down quadratic"),
        ("x^2 - 4", "quadratic shifted down 4"),
        ("x^2 + x", "quadratic with linear"),
        ("x^2 + 2*x", "quadratic with 2x"),
        ("x^2 + 3*x", "quadratic linear combo"),
        ("x^2 - x", "quadratic minus x"),
        ("x^2 - 2*x", "quadratic minus 2x"),
        ("x^2 + x + 1", "complete quadratic 1"),
        ("x^2 + 3*x + 2", "complete quadratic 2"),
        ("x^2 + 4*x + 3", "complete quadratic 3"),
        ("x^2 - 5*x + 6", "factored form quadratic"),
        ("x^2 - 3*x + 2", "factored quadratic 2"),
        ("2*x^2 + 4*x", "scaled quadratic with linear"),
        ("2*x^2 + x", "scaled quadratic plus x"),
        ("3*x^2 + 2*x", "triple quadratic linear"),
        ("-x^2 + 4*x", "negative quadratic linear"),
        ("x^2 + 4", "quadratic shifted 4"),
        ("x^2 + 10", "quadratic shifted 10"),
        ("x^2 - 9", "quadratic shifted -9"),
        ("0.25*x^2", "quarter quadratic"),
        ("1.5*x^2", "one-half quadratic"),
        ("x^2/2", "quadratic divided 2"),
        ("x^2/4", "quadratic divided 4"),
        ("-x^2 + 1", "inverted quadratic shifted"),
        ("-x^2 + 5", "inverted quadratic up 5"),
        ("-2*x^2 + 4", "inverted scaled quadratic"),
        ("2*x^2 - 3*x", "scaled quad minus linear"),
        ("3*x^2 + x + 1", "triple quad complete"),
        ("x^2 + 5*x + 6", "quad 5x plus 6"),
        ("x^2 - 4*x + 4", "perfect square"),
        ("x^2 + 6*x + 9", "perfect square 2"),
        
        # Cubic (35)
        ("x^3", "basic cubic"),
        ("2*x^3", "scaled cubic"),
        ("3*x^3", "triple cubic"),
        ("-x^3", "negative cubic"),
        ("-2*x^3", "negative scaled cubic"),
        ("0.5*x^3", "half cubic"),
        ("x^3 + 1", "cubic shifted"),
        ("x^3 - 1", "cubic shifted down"),
        ("x^3 + x", "cubic with linear"),
        ("x^3 - x", "cubic minus linear"),
        ("x^3 + 2*x", "cubic plus 2x"),
        ("x^3 - 2*x", "cubic minus 2x"),
        ("x^3 + x^2", "cubic with quadratic"),
        ("x^3 + 2*x^2", "cubic with 2x squared"),
        ("x^3 - x^2", "cubic minus x squared"),
        ("x^3 + x^2 + x", "cubic quad linear"),
        ("x^3 + x^2 - x", "cubic quad minus x"),
        ("x^3 - x^2 + x", "cubic negative quad"),
        ("x^3 + 2*x^2 + x", "cubic 2quad linear"),
        ("2*x^3 + x", "scaled cubic linear"),
        ("2*x^3 - x", "scaled cubic minus x"),
        ("x^3 + 3*x^2 + 3*x + 1", "cubic expansion"),
        ("x^3 - 3*x^2 + 3*x - 1", "negative cubic expansion"),
        ("-x^3 + x", "negative cubic plus x"),
        ("x^3/2", "cubic divided"),
        ("x^3 + 5", "cubic shifted 5"),
        ("x^3 - 8", "cubic shifted -8"),
        ("0.25*x^3", "quarter cubic"),
        ("1.5*x^3", "one-half cubic"),
        ("-x^3 + 2*x", "negative cubic linear"),
        ("2*x^3 + x^2", "scaled cubic quad"),
        ("3*x^3 - x^2", "triple cubic quad"),
        ("x^3 + 4*x^2 + 4*x", "cubic complete"),
        ("x^3 - 6*x^2 + 9*x", "cubic factored"),
        ("x^3 + 3*x^2 - x", "cubic mixed"),
        
        # Higher degree (35)
        ("x^4", "quartic function"),
        ("x^5", "quintic function"),
        ("2*x^4", "scaled quartic"),
        ("-x^4", "negative quartic"),
        ("x^4 + 1", "quartic shifted"),
        ("x^4 + x^2", "even powers"),
        ("x^4 - x^2", "difference of even powers"),
        ("x^4 + x", "quartic linear"),
        ("x^4 - x", "quartic minus x"),
        ("0.5*x^4", "half quartic"),
        ("0.25*x^4", "quarter quartic"),
        ("-2*x^4 + 4", "inverted quartic"),
        ("x^5 + x", "quintic linear"),
        ("x^5 - x", "quintic minus x"),
        ("2*x^5", "scaled quintic"),
        ("-x^5", "negative quintic"),
        ("x^5 + x^3", "quintic cubic"),
        ("x^4 + x^3", "quartic cubic combo"),
        ("x^4 - x^3", "quartic minus cubic"),
        ("x^4 + 2*x^2", "quartic 2x squared"),
        ("x^4 - 2*x^2", "quartic minus 2x squared"),
        ("x^5 + x^2", "quintic quadratic"),
        ("x^5 - x^2", "quintic minus quadratic"),
        ("x^4 + x^2 + 1", "quartic complete"),
        ("x^5 + x^3 + x", "quintic odd powers"),
        ("2*x^4 + x^2", "scaled quartic quad"),
        ("-x^4 + x^2", "negative quartic quad"),
        ("x^4/2", "quartic divided"),
        ("x^5/2", "quintic divided"),
        ("0.5*x^5", "half quintic"),
        ("-2*x^5", "negative scaled quintic"),
        ("x^4 + 4*x^2 + 4", "quartic perfect"),
        ("x^4 - 4*x^2 + 3", "quartic factored"),
        ("x^5 + 5*x^3 + 4*x", "quintic expanded"),
        ("x^6", "sixth power"),
    ],
    
    "conceptual": [
        # Trigonometric and transcendental (157 samples)
        # Basic trig (30)
        ("sin(x)", "sine function"),
        ("cos(x)", "cosine function"),
        ("tan(x)", "tangent function"),
        ("2*sin(x)", "scaled sine"),
        ("3*sin(x)", "triple sine"),
        ("-sin(x)", "negative sine"),
        ("-2*sin(x)", "negative scaled sine"),
        ("0.5*sin(x)", "half sine"),
        ("sin(x) + 1", "shifted sine up"),
        ("sin(x) + 2", "shifted sine up 2"),
        ("sin(x) - 1", "shifted sine down"),
        ("cos(x) + 1", "shifted cosine up"),
        ("cos(x) + 2", "shifted cosine up 2"),
        ("cos(x) - 1", "shifted cosine down"),
        ("2*cos(x)", "scaled cosine"),
        ("3*cos(x)", "triple cosine"),
        ("-cos(x)", "negative cosine"),
        ("-2*cos(x)", "negative scaled cosine"),
        ("0.5*cos(x)", "half cosine"),
        ("tan(x) + 1", "shifted tangent"),
        ("2*tan(x)", "scaled tangent"),
        ("-tan(x)", "negative tangent"),
        ("sin(x)/2", "sine divided"),
        ("cos(x)/2", "cosine divided"),
        ("sin(x) + 3", "sine shifted 3"),
        ("cos(x) + 3", "cosine shifted 3"),
        ("-sin(x) + 1", "negative sine shifted"),
        ("-cos(x) + 1", "negative cosine shifted"),
        ("1.5*sin(x)", "one-half sine"),
        ("1.5*cos(x)", "one-half cosine"),
        
        # Frequency variations (25)
        ("sin(2*x)", "compressed sine"),
        ("sin(3*x)", "triple frequency sine"),
        ("sin(x/2)", "stretched sine"),
        ("sin(x/3)", "third frequency sine"),
        ("cos(2*x)", "compressed cosine"),
        ("cos(3*x)", "triple frequency cosine"),
        ("cos(x/2)", "stretched cosine"),
        ("cos(x/3)", "third frequency cosine"),
        ("sin(4*x)", "quad frequency sine"),
        ("cos(4*x)", "quad frequency cosine"),
        ("sin(0.5*x)", "half frequency sine"),
        ("cos(0.5*x)", "half frequency cosine"),
        ("2*sin(2*x)", "scaled compressed sine"),
        ("2*cos(2*x)", "scaled compressed cosine"),
        ("sin(2*x) + 1", "compressed sine shifted"),
        ("cos(2*x) + 1", "compressed cosine shifted"),
        ("-sin(2*x)", "negative compressed sine"),
        ("-cos(2*x)", "negative compressed cosine"),
        ("sin(x/2) + 1", "stretched sine shifted"),
        ("cos(x/2) + 1", "stretched cosine shifted"),
        ("3*sin(3*x)", "triple scaled compressed"),
        ("tan(2*x)", "compressed tangent"),
        ("tan(x/2)", "stretched tangent"),
        ("sin(5*x)", "fifth frequency sine"),
        ("cos(5*x)", "fifth frequency cosine"),
        
        # Exponential (25)
        ("e^x", "exponential function"),
        ("2*e^x", "scaled exponential"),
        ("3*e^x", "triple exponential"),
        ("-e^x", "negative exponential"),
        ("e^x + 1", "exponential shifted"),
        ("e^x - 1", "exponential shifted down"),
        ("e^(2*x)", "fast exponential"),
        ("e^(3*x)", "triple rate exponential"),
        ("e^(x/2)", "slow exponential"),
        ("e^(-x)", "decay exponential"),
        ("e^(-2*x)", "fast decay"),
        ("2*e^(-x)", "scaled decay"),
        ("e^x/2", "exponential divided"),
        ("-2*e^x", "negative scaled exponential"),
        ("e^x + 2", "exponential shifted 2"),
        ("e^x - 2", "exponential shifted -2"),
        ("0.5*e^x", "half exponential"),
        ("1.5*e^x", "one-half exponential"),
        ("e^(x + 1)", "exponential phase shift"),
        ("e^(x - 1)", "exponential phase shift 2"),
        ("3*e^(2*x)", "triple fast exponential"),
        ("e^(-x/2)", "slow decay"),
        ("2*e^(x/2)", "scaled slow exponential"),
        ("-e^(-x)", "negative decay"),
        ("e^(0.5*x)", "half rate exponential"),
        
        # Logarithmic (20)
        ("ln(x)", "natural logarithm"),
        ("log(x)", "logarithm base 10"),
        ("2*ln(x)", "scaled logarithm"),
        ("3*ln(x)", "triple logarithm"),
        ("-ln(x)", "negative logarithm"),
        ("ln(x) + 1", "shifted logarithm"),
        ("ln(x) - 1", "log shifted down"),
        ("ln(x + 1)", "shifted log input"),
        ("ln(x + 2)", "log input shifted 2"),
        ("ln(2*x)", "log of scaled x"),
        ("ln(x/2)", "log of half x"),
        ("ln(x) + 2", "log shifted 2"),
        ("ln(x) - 2", "log shifted -2"),
        ("0.5*ln(x)", "half logarithm"),
        ("-2*ln(x)", "negative scaled log"),
        ("ln(x + 3)", "log shifted input 3"),
        ("ln(3*x)", "log of triple x"),
        ("1.5*ln(x)", "one-half logarithm"),
        ("ln(x)/2", "log divided"),
        ("2*ln(x + 1)", "scaled shifted log"),
        
        # Roots and reciprocals (27)
        ("sqrt(x)", "square root"),
        ("2*sqrt(x)", "scaled square root"),
        ("-sqrt(x)", "negative square root"),
        ("sqrt(x) + 1", "shifted square root"),
        ("sqrt(x + 1)", "shifted root input"),
        ("sqrt(x + 2)", "root input shifted 2"),
        ("sqrt(2*x)", "scaled root input"),
        ("sqrt(x/2)", "root of half x"),
        ("sqrt(x) - 1", "root shifted down"),
        ("0.5*sqrt(x)", "half square root"),
        ("1.5*sqrt(x)", "one-half square root"),
        ("sqrt(x + 3)", "root shifted 3"),
        ("sqrt(3*x)", "root of triple x"),
        ("1/x", "reciprocal function"),
        ("2/x", "scaled reciprocal"),
        ("3/x", "triple reciprocal"),
        ("-1/x", "negative reciprocal"),
        ("1/(x + 1)", "shifted reciprocal"),
        ("1/(x - 1)", "reciprocal x minus 1"),
        ("1/(2*x)", "reciprocal of 2x"),
        ("1/x^2", "inverse square"),
        ("2/x^2", "scaled inverse square"),
        ("1/x^3", "inverse cube"),
        ("-2/x", "negative scaled reciprocal"),
        ("1/(x + 2)", "reciprocal shifted 2"),
        ("2/(x + 1)", "scaled shifted reciprocal"),
        ("1/sqrt(x)", "reciprocal square root"),
        
        # Simple products (30)
        ("x*sin(x)", "product polynomial trig"),
        ("x*cos(x)", "product with cosine"),
        ("2*x*sin(x)", "scaled x sine"),
        ("x*tan(x)", "x times tangent"),
        ("-x*sin(x)", "negative x sine"),
        ("x*sin(2*x)", "x sine compressed"),
        ("x*cos(2*x)", "x cosine compressed"),
        ("x^2*sin(x)", "quadratic times sine"),
        ("x^2*cos(x)", "quadratic times cosine"),
        ("2*x^2*sin(x)", "scaled quad sine"),
        ("-x^2*sin(x)", "negative quad sine"),
        ("x^2*sin(2*x)", "quad sine compressed"),
        ("sin(x)*cos(x)", "product of trig"),
        ("sin(x)*sin(2*x)", "sine product"),
        ("cos(x)*cos(2*x)", "cosine product"),
        ("x*e^x", "linear exponential product"),
        ("2*x*e^x", "scaled x exponential"),
        ("-x*e^x", "negative x exponential"),
        ("x*e^(2*x)", "x fast exponential"),
        ("x*e^(-x)", "x decay"),
        ("x*ln(x)", "x times logarithm"),
        ("2*x*ln(x)", "scaled x log"),
        ("x^2*e^x", "quad exponential"),
        ("x*sqrt(x)", "x times root"),
        ("x/sin(x)", "x over sine"),
        ("sin(x)/cos(x)", "tangent form"),
        ("x*sin(x)*cos(x)", "x sine cosine"),
        ("x^2*tan(x)", "quad tangent"),
        ("-x*cos(x)", "negative x cosine"),
        ("x*sin(x/2)", "x stretched sine"),
    ],
    
    "application": [
        # Complex combinations (200 samples)
        # Optimization problems (30)
        ("x^3 - 3*x^2 + 2", "cubic optimization"),
        ("x^3 + 2*x^2 - 5*x", "cubic with roots"),
        ("x^3 - 6*x^2 + 9*x", "cubic critical points"),
        ("x^4 - 4*x^2", "quartic optimization"),
        ("x^4 - 8*x^2 + 16", "quartic critical"),
        ("x^3 - 12*x", "cubic simple roots"),
        ("x^3 + 3*x^2 - 9*x", "cubic inflection"),
        ("2*x^3 - 9*x^2 + 12*x", "scaled cubic opt"),
        ("x^4 - 2*x^2 + 1", "quartic perfect"),
        ("x^3 - 3*x + 2", "cubic standard form"),
        ("-x^3 + 3*x^2", "negative cubic opt"),
        ("x^4 + 4*x^3", "quartic cubic mix"),
        ("x^3 - x^2 - x + 1", "cubic factored"),
        ("x^4 - 5*x^2 + 4", "quartic factored"),
        ("2*x^3 - 3*x^2 - 12*x", "scaled cubic"),
        ("x^3 + x^2 - 2*x", "cubic roots"),
        ("x^4 - x^2 - 6", "quartic simple"),
        ("-x^3 + 6*x^2 - 9*x", "negative cubic crit"),
        ("x^3 - 4*x", "cubic two roots"),
        ("x^4 - 3*x^2 + 2", "quartic two pairs"),
        ("x^3 + 6*x^2 + 9*x", "cubic derivative"),
        ("x^4 + 2*x^2 + 1", "quartic sum"),
        ("x^3 - 9*x", "cubic simple"),
        ("2*x^3 - 6*x", "scaled cubic roots"),
        ("-x^4 + 4*x^2", "negative quartic"),
        ("x^3 + 3*x^2 + 3*x + 1", "cubic perfect"),
        ("x^4 - 4*x^3 + 4*x^2", "quartic perfect 2"),
        ("x^3 - 2*x^2 + x", "cubic factored 2"),
        ("x^4 + x^2", "quartic even simple"),
        ("x^3 - 27", "cubic shift large"),
        
        # Product rule examples (35)
        ("x^2*e^x", "product rule example"),
        ("x^3*e^x", "cubic exponential product"),
        ("x^4*e^x", "quartic exponential"),
        ("x*e^(2*x)", "x fast exponential"),
        ("x^2*e^(2*x)", "quad fast exponential"),
        ("x^2*e^(-x)", "quad decay"),
        ("x^3*e^(-x)", "cubic decay"),
        ("x^2*ln(x)", "quadratic log product"),
        ("x^3*ln(x)", "cubic log product"),
        ("x*ln(x)^2", "x log squared"),
        ("x^2*sin(x)", "quadratic sine product"),
        ("x^2*cos(x)", "quadratic cosine product"),
        ("x^3*sin(x)", "cubic trig product"),
        ("x^3*cos(x)", "cubic cosine product"),
        ("x*sin(x)*cos(x)", "triple product"),
        ("x^2*sin(2*x)", "quad compressed sine"),
        ("x^2*cos(2*x)", "quad compressed cosine"),
        ("x*cos(2*x)", "linear compressed cosine"),
        ("x*tan(x)", "x tangent product"),
        ("x^2*tan(x)", "quad tangent product"),
        ("x*e^x*sin(x)", "x exp sine"),
        ("x^2*e^x*cos(x)", "quad exp cosine"),
        ("x*ln(x)*sin(x)", "x log sine"),
        ("2*x^2*e^x", "scaled quad exp"),
        ("3*x*e^(2*x)", "scaled x fast exp"),
        ("-x^2*e^x", "negative quad exp"),
        ("x^2*e^x*sin(x)", "quad exp sine"),
        ("x*e^(-x)*cos(x)", "x decay cosine"),
        ("x^3*e^(-2*x)", "cubic fast decay"),
        ("x^2*sin(x)*cos(x)", "quad sine cosine"),
        ("x^4*ln(x)", "quartic log"),
        ("x*sqrt(x)*sin(x)", "x root sine"),
        ("x^2*sqrt(x)", "quad root"),
        ("x^3/e^x", "cubic over exp"),
        ("x^2*ln(2*x)", "quad scaled log"),
        
        # Quotient rule (30)
        ("sin(x)/x", "quotient with trig"),
        ("x/sin(x)", "reciprocal sinc"),
        ("cos(x)/x", "cosine over x"),
        ("x/cos(x)", "x over cosine"),
        ("e^x/x", "exponential over x"),
        ("x/e^x", "x over exponential"),
        ("ln(x)/x", "log over x"),
        ("x/ln(x)", "x over log"),
        ("x^2/e^x", "quadratic over exp"),
        ("e^x/x^2", "exp over quadratic"),
        ("sin(x)/cos(x)", "tangent quotient"),
        ("cos(x)/sin(x)", "cotangent"),
        ("x^2/sin(x)", "quad over sine"),
        ("sin(x)/x^2", "sine over quad"),
        ("x^3/e^x", "cubic over exp"),
        ("e^x/x^3", "exp over cubic"),
        ("tan(x)/x", "tangent over x"),
        ("x/tan(x)", "x over tangent"),
        ("x^2/ln(x)", "quad over log"),
        ("ln(x)/x^2", "log over quad"),
        ("e^x/(x + 1)", "exp over linear"),
        ("(x + 1)/e^x", "linear over exp"),
        ("sin(x)/(x + 1)", "sine over linear"),
        ("x^2/(x + 1)", "quad over linear"),
        ("x^3/(x^2 + 1)", "cubic over quad plus"),
        ("(x^2 + 1)/x", "quad plus over x"),
        ("x/(x^2 + 1)", "rational with denom"),
        ("x^2/(x^2 + 1)", "bounded rational"),
        ("e^x/(x^2 + 1)", "exp over quad plus"),
        ("sin(x)/(x^2 + 1)", "sine over quad plus"),
        
        # Composition (35)
        ("sin(x^2)", "sine of square"),
        ("cos(x^2)", "cosine of square"),
        ("e^(sin(x))", "exponential of sine"),
        ("e^(cos(x))", "exponential of cosine"),
        ("ln(sin(x))", "log of sine"),
        ("ln(cos(x))", "log of cosine"),
        ("sin(e^x)", "sine of exponential"),
        ("cos(e^x)", "cosine of exponential"),
        ("sin(ln(x))", "sine of log"),
        ("cos(ln(x))", "cosine of log"),
        ("e^(x^2)", "exp of square"),
        ("ln(x^2)", "logarithm composition"),
        ("ln(x^2 + 1)", "log of sum"),
        ("ln(x^2 + 2)", "log of quad plus 2"),
        ("e^(2*x^2)", "exp of scaled square"),
        ("sin(sqrt(x))", "sine of root"),
        ("cos(sqrt(x))", "cosine of root"),
        ("sqrt(sin(x))", "root of sine"),
        ("sqrt(cos(x))", "root of cosine"),
        ("sqrt(e^x)", "root of exponential"),
        ("sqrt(ln(x))", "root of log"),
        ("ln(e^x + 1)", "log exp sum"),
        ("e^(ln(x))", "exp of log"),
        ("sin(2*x^2)", "sine of scaled square"),
        ("cos(3*x^2)", "cosine triple square"),
        ("e^(x^2 + x)", "exp of quad linear"),
        ("ln(x^2 + x)", "log of quad linear"),
        ("sin(x^3)", "sine of cube"),
        ("cos(x^3)", "cosine of cube"),
        ("e^(x^3)", "exp of cube"),
        ("ln(x^3)", "log of cube"),
        ("sqrt(x^2 + 1)", "root of quad plus"),
        ("sqrt(x^2 + 4)", "shifted radical composition"),
        ("sqrt(1 + x^2)", "composition with radical"),
        ("ln(sqrt(x))", "log of root"),
        
        # Combinations (30)
        ("sin(x) + cos(x)", "sum of trig"),
        ("sin(x) - cos(x)", "difference of trig"),
        ("e^x + e^(-x)", "hyperbolic cosine"),
        ("e^x - e^(-x)", "hyperbolic sine"),
        ("sin(x) + x", "sine plus x"),
        ("cos(x) + x", "cosine plus x"),
        ("e^x + x", "exp plus x"),
        ("ln(x) + x", "log plus x"),
        ("sin(x) + e^x", "sine plus exp"),
        ("cos(x) + ln(x)", "cosine plus log"),
        ("x^2 + sin(x)", "quad plus sine"),
        ("x^2 + e^x", "quad plus exp"),
        ("x^3 + ln(x)", "cubic plus log"),
        ("sin(x)*e^x", "sine exponential product"),
        ("cos(x)*e^x", "cosine exponential product"),
        ("sin(x)*ln(x)", "sine log product"),
        ("cos(x)*ln(x)", "cosine log product"),
        ("tan(x)*x", "tangent linear product"),
        ("e^(-x)*sin(x)", "damped oscillation"),
        ("e^(-x)*cos(x)", "damped cosine"),
        ("x*e^(-x)", "scaled decay"),
        ("x^2*e^(-x)", "quadratic decay"),
        ("x*sin(x) + cos(x)", "x sine plus cosine"),
        ("x*cos(x) - sin(x)", "x cosine minus sine"),
        ("e^x*sin(x)", "exp sine product"),
        ("e^x*cos(x)", "exp cosine product"),
        ("ln(x)*sin(x)", "log sine product"),
        ("sqrt(x)*sin(x)", "root sine product"),
        ("x^2 + x*sin(x)", "quad plus x sine"),
        ("x^3 - x*cos(x)", "cubic minus x cosine"),
        
        # Special functions (40)
        ("e^(-x^2)", "gaussian function"),
        ("x*e^(-x)", "scaled decay product"),
        ("x^2*e^(-x)", "quadratic decay product"),
        ("x*e^(-x^2)", "gaussian with linear"),
        ("x^2*e^(-x^2)", "gaussian with quadratic"),
        ("x^3*e^(-x)", "cubic decay"),
        ("x^4*e^(-x)", "quartic decay"),
        ("e^(-2*x)*sin(x)", "fast decay oscillation"),
        ("e^(-x)*sin(2*x)", "decay compressed sine"),
        ("e^(-x)*cos(2*x)", "decay compressed cosine"),
        ("x*e^(-2*x)", "x fast decay"),
        ("x^2*e^(-2*x)", "quad fast decay"),
        ("sin(x)^2", "sine squared"),
        ("cos(x)^2", "cosine squared"),
        ("e^(x)*x^2", "exp quad product alt"),
        ("ln(x + 1)", "shifted logarithm"),
        ("ln(2*x + 1)", "scaled shifted log"),
        ("e^(x + 1)", "shifted exponential"),
        ("sin(x + 1)", "phase shifted sine"),
        ("cos(x - 1)", "phase shifted cosine"),
        ("sqrt(x)*e^(-x)", "root decay"),
        ("sqrt(x)*ln(x)", "root log"),
        ("x/sqrt(x)", "x over root"),
        ("x^2/sqrt(x)", "quad over root"),
        ("sin(x)/sqrt(x)", "sine over root"),
        ("e^x/sqrt(x)", "exp over root"),
        ("ln(x)/sqrt(x)", "log over root"),
        ("x*ln(x + 1)", "x shifted log"),
        ("x^2*ln(x + 1)", "quad shifted log"),
        ("e^(x)*ln(x)", "exp log product"),
        ("sin(x)*cos(2*x)", "sine cosine different freq"),
        ("sin(2*x)*cos(x)", "compressed sine cosine"),
        ("x^2*sin(x/2)", "quad stretched sine"),
        ("x*cos(x/2)", "x stretched cosine"),
        ("e^(x/2)*sin(x)", "slow exp sine"),
        ("e^(2*x)*cos(x)", "fast exp cosine"),
        ("ln(x)*cos(x)", "log cosine product"),
        ("ln(x^2)*sin(x)", "log squared sine"),
        ("x*ln(x)*cos(x)", "x log cosine"),
        ("sqrt(x)*cos(x)", "root cosine"),
        ("sqrt(x)*e^x", "root exp"),
    ],
    
    "advanced": [
        # Complex multi-concept (50 samples to reach 537 total)
        # Advanced compositions (20)
        ("e^(sin(x))", "nested transcendental"),
        ("e^(cos(x))", "exponential of cosine"),
        ("sin(e^x)", "sine of exponential"),
        ("cos(e^x)", "cosine of exponential"),
        ("e^(tan(x))", "exp of tangent"),
        ("tan(e^x)", "tangent of exp"),
        ("ln(sin(x))", "log of sine"),
        ("ln(cos(x))", "log of cosine"),
        ("ln(tan(x))", "log of tangent"),
        ("sin(ln(x))", "sine of log"),
        ("cos(ln(x))", "cosine of log"),
        ("tan(ln(x))", "tangent of log"),
        ("e^(x^2)*sin(x)", "exp square sine"),
        ("e^(x^2)*cos(x)", "exp square cosine"),
        ("sin(x^2)*cos(x)", "sine square cosine"),
        ("ln(x^2 + 2*x + 1)", "log of complete square"),
        ("e^(sqrt(x))", "exp of root"),
        ("sin(sqrt(x))", "sine of root"),
        ("ln(ln(x))", "double logarithm"),
        ("e^(e^x)", "double exponential"),
        
        # Advanced products (15)
        ("x^3*sin(x)*cos(x)", "cubic sine cosine"),
        ("x^2*e^x*sin(x)", "quad exp sine"),
        ("x*e^x*ln(x)", "x exp log"),
        ("x^2*ln(x)*sin(x)", "quad log sine"),
        ("x*sin(x)*ln(x)", "x sine log"),
        ("e^x*ln(x)*cos(x)", "exp log cosine"),
        ("x^3*e^(-x)", "cubic decay advanced"),
        ("x^4*e^(-x)", "quartic decay advanced"),
        ("x^2*e^(-x)*sin(x)", "quad decay sine"),
        ("x*e^(-x)*cos(x)", "x decay cosine"),
        ("x^3*ln(x)*cos(x)", "cubic log cosine"),
        ("x^2*sqrt(x)*sin(x)", "quad root sine"),
        ("x*e^x*cos(2*x)", "x exp compressed cosine"),
        ("x^2*sin(x)*e^(-x)", "quad sine decay"),
        ("x^3*cos(x)*ln(x)", "cubic cosine log"),
        
        # Complex rational (15)
        ("x^2/(1 + x^2)", "rational with asymptote"),
        ("x^3/(1 + x^2)", "cubic over quadratic"),
        ("(x^2 - 1)/(x^2 + 1)", "complex rational"),
        ("(x^3 + 1)/(x^2 + 1)", "cubic over quadratic 2"),
        ("(x^2 + x)/(x^2 + 1)", "quad linear over quad"),
        ("(x^3 - x)/(x^2 + 1)", "cubic linear over quad"),
        ("x/(x^3 + 1)", "x over cubic plus"),
        ("x^2/(x^3 + 1)", "quad over cubic plus"),
        ("(x^2 + 1)/(x^3 + 1)", "quad plus over cubic plus"),
        ("(x^4 - 1)/(x^2 + 1)", "quartic over quad plus"),
        ("(2*x^2 + x)/(x^2 + 1)", "scaled quad over quad plus"),
        ("(x^3 + 2*x)/(x^2 + 4)", "cubic linear over quad 4"),
        ("x^2/(x^2 + 4)", "quad over quad plus 4"),
        ("(x^3 - x^2)/(x + 1)", "cubic quad over linear"),
        ("(x^4 + x^2)/(x^2 + 1)", "quartic over quad plus"),
        
        # Advanced combinations (remaining samples)
        ("sin(x^2)*e^x", "sine square exp"),
        ("cos(x^2)*ln(x)", "cosine square log"),
        ("e^(sin(x))*x", "exp sine times x"),
        ("ln(sin(x))*x", "log sine times x"),
        ("x*e^(cos(x))", "x exp cosine"),
        ("sqrt(sin(x))*x", "root sine times x"),
        ("sqrt(cos(x))*ln(x)", "root cosine log"),
        ("sqrt(e^x)*sin(x)", "root exp sine"),
        ("x^2*e^(sin(x))", "quad exp sine nested"),
        ("x*ln(sin(x))", "x log sine"),
        ("e^x*sin(x^2)", "exp sine square"),
        ("ln(x)*cos(x^2)", "log cosine square"),
        ("sin(x)*cos(x)*tan(x)", "triple trig product"),
        ("x*sin(x)*cos(x)*e^x", "quad product advanced"),
        ("(sin(x) + cos(x))/x", "trig sum over x"),
        ("x/(sin(x) + cos(x))", "x over trig sum"),
        ("e^x/(sin(x) + 1)", "exp over sine plus"),
        ("ln(x)/(cos(x) + 1)", "log over cosine plus"),
        ("x^2*sin(x)/cos(x)", "quad sine over cosine"),
        ("e^x*sin(x)/x", "exp sine over x"),
    ]
}

class DatasetGenerator:
    def __init__(self, output_dir: str = "derivative_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.code_dir = self.output_dir / "code"
        self.metadata_dir = self.output_dir / "metadata"
        self.code_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Progress tracking files
        self.progress_file = self.metadata_dir / "progress.json"
        self.checkpoint_file = self.metadata_dir / "checkpoint.json"
        
        self.results = []
        self.start_time = datetime.now()
        
        # Load existing progress if available
        self._load_progress()
        
    def _load_progress(self):
        """Load existing progress from previous runs"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    self.results = data.get("results", [])
                    print(f"\n✅ Loaded {len(self.results)} previous results")
                    
                    # Show what's already done
                    successful = sum(1 for r in self.results if r["success"])
                    print(f"   • {successful} successful")
                    print(f"   • {len(self.results) - successful} failed")
            except Exception as e:
                print(f"⚠️ Could not load progress: {e}")
                self.results = []
    
    def _get_completed_functions(self) -> set:
        """Get set of functions that have been completed"""
        return {r["function"] for r in self.results}
    
    def _save_checkpoint(self, level: str, function_index: int):
        """Save checkpoint for resuming"""
        with open(self.checkpoint_file, 'w') as f:
            json.dump({
                "last_level": level,
                "last_index": function_index,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
    
    def _load_checkpoint(self) -> Optional[tuple]:
        """Load checkpoint if exists"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    return data.get("last_level"), data.get("last_index", -1)
            except:
                return None
        return None
    
    def generate_dataset(
        self, 
        levels: List[str] = None, 
        delay_between: float = 2.0,
        daily_limit: int = 240,  # Leave buffer below 250 RPD limit
        resume: bool = True
    ):
        """Generate complete dataset with multi-day support
        
        Args:
            levels: List of levels to generate, default is all
            delay_between: Seconds to wait between generations
            daily_limit: Max generations per day (240 to stay under 250 RPD)
            resume: Whether to resume from previous checkpoint
        """
        if levels is None:
            levels = list(DATASET_FUNCTIONS.keys())
        
        # Get completed functions
        completed = self._get_completed_functions()
        
        # Load checkpoint if resuming
        checkpoint_level, checkpoint_index = None, -1
        if resume:
            checkpoint_data = self._load_checkpoint()
            if checkpoint_data:
                checkpoint_level, checkpoint_index = checkpoint_data
                print(f"\n📍 Resuming from: {checkpoint_level}, index {checkpoint_index}")
        
        total_functions = sum(len(DATASET_FUNCTIONS[level]) for level in levels)
        current = len(self.results)
        today_count = 0
        consecutive_failures = 0
        
        print(f"\n{'='*70}")
        print(f"GENERATING DATASET - DAY {len(self.results)//daily_limit + 1}")
        print(f"{'='*70}")
        print(f"Total target: {total_functions}")
        print(f"Already completed: {len(completed)}")
        print(f"Remaining: {total_functions - len(completed)}")
        print(f"Today's limit: {daily_limit} generations")
        print(f"{'='*70}\n")
        
        # Find starting point
        start_processing = checkpoint_level is None
        
        for level in levels:
            # Skip levels until we reach checkpoint
            if not start_processing:
                if level != checkpoint_level:
                    continue
                else:
                    start_processing = True
            
            print(f"\n{'='*70}")
            print(f"LEVEL: {level.upper()}")
            print(f"Functions in this level: {len(DATASET_FUNCTIONS[level])}")
            print(f"{'='*70}")
            
            level_dir = self.code_dir / level
            level_dir.mkdir(exist_ok=True)
            
            for idx, (func, description) in enumerate(DATASET_FUNCTIONS[level]):
                # Skip if before checkpoint
                if level == checkpoint_level and idx <= checkpoint_index:
                    continue
                
                # Skip if already completed
                if func in completed:
                    print(f"\n[{current+1}/{total_functions}] ⏭️  SKIPPING (already done): {func}")
                    current += 1
                    continue
                
                # Check daily limit
                if today_count >= daily_limit:
                    print(f"\n{'='*70}")
                    print(f"🌙 DAILY LIMIT REACHED ({daily_limit} generations)")
                    print(f"{'='*70}")
                    print(f"Progress: {current}/{total_functions} ({current/total_functions*100:.1f}%)")
                    print(f"Successful today: {sum(1 for r in self.results[-today_count:] if r.get('success'))}/{today_count}")
                    print(f"\nResume tomorrow by running the same command.")
                    print(f"Progress is automatically saved!")
                    self._save_progress()
                    self._save_checkpoint(level, idx - 1)
                    return
                
                current += 1
                today_count += 1
                
                print(f"\n[{current}/{total_functions}] Processing: f(x) = {func}")
                print(f"Description: {description}")
                print(f"Today's count: {today_count}/{daily_limit}")
                print("-" * 70)
                
                # Rate limiting delay
                if today_count > 1:
                    time.sleep(delay_between)
                
                # Generate code
                try:
                    code, metadata = generate_manim_code(
                        func, 
                        max_attempts=3, 
                        use_thinking=False,  # CHANGED: Don't use thinking mode (uses up 2.0-flash-exp quota)
                        skip_execution_test=False,
                        model_name="gemini-2.5-flash"  # Use model with 250 RPD limit
                    )
                    
                    # Record result
                    result = {
                        "function": func,
                        "description": description,
                        "level": level,
                        "success": bool(code),
                        "attempts": metadata.get("attempts", 0),
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata,
                        "code_length": len(code) if code else 0,
                    }
                    
                    if code:
                        # Save code
                        safe_name = func.replace("^", "pow").replace("*", "mul").replace("/", "div")
                        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in safe_name)
                        code_file = level_dir / f"{safe_name}.py"
                        
                        with open(code_file, 'w') as f:
                            f.write(code)
                        
                        result["code_file"] = str(code_file.relative_to(self.output_dir))
                        print(f"✓ SUCCESS - Saved to {code_file.name}")
                        consecutive_failures = 0
                    else:
                        print(f"✗ FAILED after {metadata.get('attempts', 0)} attempts")
                        if metadata.get('test_errors'):
                            print(f"  Last error: {metadata['test_errors'][-1][:100]}...")
                        consecutive_failures += 1
                        
                        # If too many consecutive failures, pause longer
                        if consecutive_failures >= 3:
                            print(f"⚠️ {consecutive_failures} consecutive failures. Pausing 30s...")
                            time.sleep(30)
                    
                    self.results.append(result)
                    
                    # Save progress every 10 samples
                    if current % 10 == 0:
                        self._save_progress()
                        self._save_checkpoint(level, idx)
                        self._print_interim_stats(current, total_functions, today_count, daily_limit)
                
                except Exception as e:
                    error_str = str(e)
                    print(f"❌ CRITICAL ERROR: {error_str[:200]}")
                    
                    # Check if rate limit error
                    if "429" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower():
                        print(f"\n{'='*70}")
                        print(f"⚠️ API RATE LIMIT REACHED")
                        print(f"{'='*70}")
                        print(f"Generated today: {today_count}")
                        print(f"Progress: {current}/{total_functions}")
                        print(f"\nSaving progress and stopping for today...")
                        
                        # Don't count this failed attempt
                        self._save_progress()
                        self._save_checkpoint(level, idx - 1)
                        return
                    
                    # Log error and continue
                    result = {
                        "function": func,
                        "description": description,
                        "level": level,
                        "success": False,
                        "attempts": 0,
                        "timestamp": datetime.now().isoformat(),
                        "metadata": {"error": error_str[:500]},
                        "code_length": 0,
                    }
                    self.results.append(result)
                    time.sleep(10)
        
        # All done!
        print(f"\n{'='*70}")
        print("🎉 ALL FUNCTIONS PROCESSED!")
        print(f"{'='*70}")
        self._print_summary()
        self._save_final_report()
        
    def _print_interim_stats(self, current: int, total: int, today_count: int, daily_limit: int):
        """Print interim statistics"""
        successful = sum(1 for r in self.results if r["success"])
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = today_count / elapsed if elapsed > 0 else 0
        remaining_today = daily_limit - today_count
        eta_seconds = remaining_today / rate if rate > 0 else 0
        
        print(f"\n{'─'*70}")
        print(f"Overall Progress: {current}/{total} ({current/total*100:.1f}%)")
        print(f"Today's Progress: {today_count}/{daily_limit} ({today_count/daily_limit*100:.1f}%)")
        print(f"Success Rate: {successful}/{current} ({successful/current*100:.1f}%)")
        print(f"Rate: {rate*60:.1f} samples/minute")
        if remaining_today > 0:
            print(f"ETA for today: {eta_seconds/60:.1f} minutes")
        print(f"{'─'*70}")
    
    def _save_progress(self):
        """Save current progress to JSON"""
        with open(self.progress_file, 'w') as f:
            json.dump({
                "last_updated": datetime.now().isoformat(),
                "total_processed": len(self.results),
                "results": self.results
            }, f, indent=2)
        print(f"💾 Progress saved ({len(self.results)} results)")
    
    def _print_summary(self):
        """Print generation summary"""
        total = len(self.results)
        successful = sum(1 for r in self.results if r["success"])
        failed = total - successful
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        print(f"\n{'='*70}")
        print("DATASET GENERATION SUMMARY")
        print(f"{'='*70}")
        print(f"Total Functions Processed: {total}")
        print(f"Successful: {successful} ({successful/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")
        print(f"Session Time: {elapsed/60:.1f} minutes")
        
        # By level
        print("\n📊 Success Rate by Level:")
        for level in DATASET_FUNCTIONS.keys():
            level_results = [r for r in self.results if r["level"] == level]
            if level_results:
                level_success = sum(1 for r in level_results if r["success"])
                print(f"  {level.capitalize():12} {level_success:3}/{len(level_results):3} ({level_success/len(level_results)*100:5.1f}%)")
    
    def _save_final_report(self):
        """Save comprehensive report"""
        # JSON report
        report_json = self.metadata_dir / "generation_report.json"
        with open(report_json, 'w') as f:
            json.dump({
                "generation_date": datetime.now().isoformat(),
                "total_time_minutes": (datetime.now() - self.start_time).total_seconds() / 60,
                "summary": {
                    "total": len(self.results),
                    "successful": sum(1 for r in self.results if r["success"]),
                    "failed": sum(1 for r in self.results if not r["success"]),
                    "success_rate": sum(1 for r in self.results if r["success"]) / len(self.results) * 100 if self.results else 0,
                },
                "by_level": {
                    level: {
                        "total": len([r for r in self.results if r["level"] == level]),
                        "successful": len([r for r in self.results if r["level"] == level and r["success"]]),
                    }
                    for level in DATASET_FUNCTIONS.keys()
                },
                "results": self.results
            }, f, indent=2)
        
        # CSV report
        report_csv = self.metadata_dir / "generation_report.csv"
        with open(report_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "function", "description", "level", "success", 
                "attempts", "code_length", "code_file"
            ])
            writer.writeheader()
            for r in self.results:
                writer.writerow({
                    "function": r["function"],
                    "description": r["description"],
                    "level": r["level"],
                    "success": r["success"],
                    "attempts": r["attempts"],
                    "code_length": r["code_length"],
                    "code_file": r.get("code_file", "")
                })
        
        print(f"\n📁 Reports saved:")
        print(f"  • {report_json}")
        print(f"  • {report_csv}")

def main():
    load_dotenv()
    
    print("=" * 70)
    print("MANIM DERIVATIVE DATASET GENERATOR")
    print("Multi-Day Generation with Automatic Resuming")
    print("=" * 70)
    
    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("\n❌ ERROR: GEMINI_API_KEY not found in environment")
        print("Please set it in your .env file")
        return
    
    # Initialize generator
    generator = DatasetGenerator(output_dir="derivative_dataset_537")
    
    print("\n📋 Configuration:")
    print("  • Model: gemini-2.5-flash (250 RPD limit)")
    print("  • Daily limit: 240 generations (with buffer)")
    print("  • Auto-resume: Enabled")
    print("  • Progress tracking: Enabled")
    print("  • Thinking mode: Disabled (saves quota)")
    
    # Generate dataset - will automatically resume from where it left off
    generator.generate_dataset(
        delay_between=2.0,
        daily_limit=240,  # Process 240 per day, stay under 250 limit
        resume=True
    )
    
    print("\n✅ Session complete!")
    print(f"📂 Check {generator.output_dir} for results")
    print("\nTo continue tomorrow, simply run this script again!")

if __name__ == "__main__":
    main()