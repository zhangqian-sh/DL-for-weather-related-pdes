#=
Modified from https://github.com/ryarazi/ExactRiemannProblemSolver, Roy Arazi
MIT License

Copyright (c) 2018 Roy Arazi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
=#

# Modified data: 2021/8/21
#Type Define:
#struct for saving the state of the system in some half
struct HydroStatus
	rho::Float64 #density
	u::Float64 #velocity
	p::Float64 #pressure
	gamma::Float64 #adiabatic index
	c::Float64 #speed of cound
	A::Float64 #Toro constant (4.8)
	B::Float64 #Toro constant (4.8)

	function HydroStatus(rho, u, p, gamma)
		c = sqrt(gamma * p / rho)
		A = 2. / (gamma + 1.) / rho
		B = (gamma - 1.) / (gamma + 1.) * p
		new(rho, u, p, gamma, c, A, B)
	end
end

#like the HydroStatus but saves only the important parameters - density, pressure and velocity
struct HydroStatusSimple
	rho::Float64 #density
	u::Float64 #velocity
	p::Float64 #pressure
end
HydroStatusSimple(status::HydroStatus) = HydroStatusSimple(status.rho, status.u, status.p)

#different types to dispatch the first guess of p_star calculation
abstract type FirstGuessScheme end
struct TwoRarefaction <: FirstGuessScheme end
struct PrimitiveValue <: FirstGuessScheme end
struct TwoShock <: FirstGuessScheme end
struct MeanPressure <: FirstGuessScheme end

#the type of the wave that will be created on some side of the contact discontinuity
abstract type WaveType end
struct Rarefaction <: WaveType end
struct Shock <: WaveType end
struct Vaccum <: WaveType end #for vaccum generated in the middle

#the side in which the calculation is happening
abstract type Side end
struct LeftSide <: Side end
struct RightSide <: Side end

#in some equations there is factor difference between the left and right side calculations
side_factor(::LeftSide) = -1
side_factor(::RightSide) = 1

#choose the some variable based on the side given
choose_side(left, right, ::LeftSide) = left
choose_side(left, right, ::RightSide) = right

# Initial guess of the p_star in iteration

# (1) Two-Rarefaction approximation, see (4.46) from Toro
function p_guess(left::HydroStatus, right::HydroStatus, ::TwoRarefaction)
		@assert left.gamma == right.gamma
		gamma = left.gamma
		gamma_power = (gamma-1) / 2. / gamma
		delta_u = right.u - left.u
		return ( (left.c + right.c - 0.5 * (gamma - 1) * delta_u) /
			   (left.c/left.p^gamma_power + right.c/right.p^gamma_power) ) ^
		       (1. / gamma_power)
end
# (2) Linearised solution based on primitive variables, see (4.47) from Toro
function p_guess(left::HydroStatus, right::HydroStatus, ::PrimitiveValue)
	delta_u = right.u - left.u
	mean_pressure = 0.5 * (left.p + right.p)
	mean_density = 0.5 * (left.rho + right.rho)
	mean_speed_of_sound = 0.5 * (left.c + right.c)
	return mean_pressure - 0.5 * delta_u * mean_density * mean_speed_of_sound
end
# (3) Two-Shock approximation, see (4.48) from Toro
function p_guess(left::HydroStatus, right::HydroStatus, ::TwoShock)
	p_hat = p_guess(left, right, PrimitiveValue())
	g_left = sqrt(left.A / (p_hat + left.B))
	g_right = sqrt(right.A / (p_hat + right.B))
	delta_u = right.u - left.u
	return (g_left * left.p + g_right * right.p - delta_u)/(g_left+g_right)
end
# (4) Arithmetic mean, see (4.49) from Toro
p_guess(left::HydroStatus, right::HydroStatus, ::MeanPressure) = 0.5*(left.p + right.p)

# (5) to provide a guess value for pressure p_star in the Star Region.
# The choice is made according to adaptive Riemann solver using
# the PrimitiveValue, TwoRarefaction and TwoShock approximate Riemann solvers.
# See Sect. 9.5 of Chapt. 9 of Ref. 1
function p_guess(left::HydroStatus, right::HydroStatus)
    Quser=2.0
    PPV=max(0.0, p_guess(left,right,PrimitiveValue()))
    Pmin=min(left.p, right.p)
    Pmax=max(left.p, right.p)
    Qmax=Pmax/Pmin
    if (Qmax≤Quser)&&(Pmin≤PPV≤Pmax)
        return PPV
    else
        if PPV<Pmin # Select Two-Rarefaction Riemann solver
            return p_guess(left,right,TwoRarefaction())
        else # Select Two-Shock Riemann solver with PVRS as estimate
            return p_guess(left,right,TwoShock())
        end
    end
end

# Riemann Solver:

# function (4.6) and (4.7) from Toro
f(p, status::HydroStatus, ::Shock) = (p - status.p) * sqrt(status.A / (p + status.B))
f(p, status::HydroStatus, ::Rarefaction) = 2. * status.c / (status.gamma - 1.) * ((p / status.p)^((status.gamma - 1.) / 2. /status.gamma) - 1.)
function f(p, status::HydroStatus)
	wave_type = p > status.p ? Shock() : Rarefaction()
	return f(p, status, wave_type)
end
# function (4.37) from Toro
f′(p, status::HydroStatus, ::Shock) = sqrt(status.A/(status.B+p))*(1-(p-status.p)/(2(status.B+p)))
f′(p, status::HydroStatus, ::Rarefaction) = 1/(status.rho*status.c)*(p/status.p)^(-(status.gamma+1)/2/status.gamma)
function f′(p, status::HydroStatus)
	wave_type = p > status.p ? Shock() : Rarefaction()
	return f′(p, status, wave_type)
end
#solver for p_star based on the first guess of the pressure
#uses Newton iteration method to solve the equation
function p_star_calc(left::HydroStatus, right::HydroStatus, TOL::Float64)
	p0 = max(TOL, p_guess(left, right))
	p=p0
	delta_u = right.u - left.u
	MAXIter=100
	I=0
	while true
		p=p0-(f(p0,left)+f(p0,right)+delta_u)/(f′(p0,left)+f′(p0,right))
		if 2.0*abs((p-p0)/(p+p0))≤TOL
			break
		end
		if p<0
			p=TOL
		end
		if I>MAXIter
			error("exceed the max num of iter")
		end
		p0=p
		I+=1
	end
	return p
end
# equation (4.9) from Toro
u_star_calc(p_star, left::HydroStatus, right::HydroStatus) = 0.5 * (left.u + right.u) +
															 0.5 * (f(p_star, right) - f(p_star, left))
# equations (4.53) and (4.60) from Toro
rho_star_calc(p_star, status::HydroStatus, ::Rarefaction) = status.rho * (p_star / status.p) ^ (1. / status.gamma)
#equations (4.50) and (4.57) from Toro
function rho_star_calc(p_star, status::HydroStatus, ::Shock)
	gamma_ratio = (status.gamma - 1.) / (status.gamma + 1.)
	pressure_ratio = p_star / status.p
	return status.rho * (pressure_ratio + gamma_ratio) / (pressure_ratio * gamma_ratio + 1.)
end
#for vaccum generation the density in the middle is zero (= vaccum)
rho_star_calc(p_star, status::HydroStatus, ::Vaccum) = 0.
# equation (4.25) and (4.34) from Toro
c_star_calc(p_star, status::HydroStatus, ::Rarefaction) = status.c * (p_star / status.p) ^ ((status.gamma - 1.) / 2. / status.gamma)
#equation (4.55) and (4.62) from Toro
head_speed_calc(p_star, u_star, status::HydroStatus,
				side::T, ::Rarefaction) where {T<:Side} = status.u + side_factor(side) * status.c

#equation (4.55) and (4.62) from Toro
function tail_speed_calc(p_star, u_star, status::HydroStatus, side::T, ::Rarefaction) where {T<:Side}
	c_star = c_star_calc(p_star, status, Rarefaction())
	return u_star + side_factor(side) * c_star
end
#equations (4.52) and (4.59) from Toro
function head_speed_calc(p_star, u_star, status::HydroStatus, side::T, ::Shock) where {T<:Side}
	gamma = status.gamma
	gamma_ratio1 = (gamma + 1.) / 2. / gamma
	gamma_ratio2 = (gamma - 1.) / 2. / gamma
	return status.u + side_factor(side) * status.c * sqrt(gamma_ratio1 * p_star / status.p + gamma_ratio2)
end
#for shock the head and the tail are the same
tail_speed_calc(p_star, u_star, status::HydroStatus, side::T, ::Shock) where {T<:Side} = head_speed_calc(p_star, u_star, status, side, Shock())

#those are the velocities of the head and the tail of the rarefaciton wave when a vaccum presents
#see (4.76), (4.77), (4.79) and (4.80) from Toro
head_speed_calc(p_star, u_star, status::HydroStatus,
				side::T, ::Vaccum) where {T<:Side} = status.u + side_factor(side) * status.c
tail_speed_calc(p_star, u_star, status::HydroStatus, side::T, ::Vaccum) where {T<:Side} = status.u - side_factor(side) * 2 * status.c / (status.gamma - 1.)

#this is the density, velocity and pressure profile of a rarefaction wave in some specific x/t
#see (4.56) and (4.63) from Toro
function rarefaction_profile(x_over_t, status::HydroStatus, side::T) where {T<:Side}
	gamma = status.gamma
	gamma_ratio = (gamma - 1.) / (gamma + 1.)
	gamma_plus = 2. / (gamma + 1.)
	gamma_minus = 2. / (gamma - 1.)

	rarefaction_factor = (gamma_plus - side_factor(side) * gamma_ratio / status.c * (status.u - x_over_t)) ^ gamma_minus

	rho = status.rho * rarefaction_factor
	u = gamma_plus * (-side_factor(side)*status.c + (1. /gamma_minus) * status.u + x_over_t)
	p = status.p * rarefaction_factor ^ gamma

	return HydroStatusSimple(rho, u, p)
end

# this is the MAIN function that samples the Riemann problem and returns the density, pressure and velocity in the space points
#specified by the array x. It gets the following parameters:
# x - an array type which saves all the points in space in which we want to sample the problem (pay attention - the contact discontinuity begins at x=0.)
# t - a floating number which tells us in what time we sample to problem
# left + right - struct from type HydroStatus (see "TypeDefine.jl") to get the initial state of the system
# TOL - the tolerance for the convergance of the p_star finding algorithm
function RiemannSolver(x, t::Float64,left::HydroStatus,right::HydroStatus;TOL::Float64 = 1.e-6)
	@assert left.gamma == right.gamma
	if left.rho == 0 #material is on the right, vaccum on the left
		return sample_riemann_side_vaccum(x, t, right, RightSide())
	elseif right.rho == 0 #material is on the left, vaccum on the right
		return sample_riemann_side_vaccum(x, t, left, LeftSide())
	else
		return sample_riemann_regular(x, t, left, right, TOL)
	end
end
function RiemannSolver(x, t::Float64,left_rho::Float64,left_u::Float64,left_p::Float64,left_gamma::Float64,
									 right_rho::Float64,right_u::Float64,right_p::Float64,right_gamma::Float64;
									 TOL::Float64 = 1.e-6)
	left=HydroStatus(left_rho,left_u,left_p,left_gamma)
	right=HydroStatus(right_rho,right_u,right_p,right_gamma)
	return RiemannSolver(x, t,left,right;TOL=TOL)
end

#sample the riemann problem in the situation where both sides are not vaccum (but vaccum can be generated through the dynamics)
function sample_riemann_regular(x, t::Float64, left::HydroStatus, right::HydroStatus, TOL::Float64)
	@assert left.rho != 0.
	@assert right.rho != 0.

	profile = similar(x, HydroStatusSimple) #we return the values of rho, u and p in each point x in space

	#the status "far" from the center = like the starting condition
	status_left = HydroStatusSimple(left)
	status_right = HydroStatusSimple(right)

	x_over_t = x ./ t

	if 2. / (left.gamma - 1.) * (left.c + right.c) <= right.u - left.u #vaccum generation
		#if vaccum is generated - the density in the middle will be zero
		p_star = 0.
		u_star = 0.
		wave_type_left = Vaccum()
		wave_type_right = Vaccum()
	else
		#if vaccum is not generated we can calculate the star profile in the middle
		p_star = p_star_calc(left, right, TOL)
		u_star = u_star_calc(p_star, left, right)
		#check what kind of waves are in every direction (shock or rarefaction)
		wave_type_left = p_star > left.p ? Shock() : Rarefaction()
		wave_type_right = p_star > right.p ? Shock() : Rarefaction()
	end

	rho_star_left = rho_star_calc(p_star, left, wave_type_left)
	status_left_star = HydroStatusSimple(rho_star_left, u_star, p_star) #the profile in the left near the contact discontinuity
	head_speed_left = head_speed_calc(p_star, u_star, left, LeftSide(), wave_type_left)
	tail_speed_left = tail_speed_calc(p_star, u_star, left, LeftSide(), wave_type_left)

	rho_star_right = rho_star_calc(p_star, right, wave_type_right)
	status_right_star = HydroStatusSimple(rho_star_right, u_star, p_star) #the profile in the right near the contact discontinuity
	head_speed_right = head_speed_calc(p_star, u_star, right, RightSide(), wave_type_right)
	tail_speed_right = tail_speed_calc(p_star, u_star, right, RightSide(), wave_type_right)


	for i = 1:length(x)
		S = x_over_t[i] #this is like the S which Toro use in Section 4.5

		#see Figure 4.14 in Toro for the flow of the following lines
		side = S < u_star ? LeftSide() : RightSide()
		status = choose_side(left, right, side)
		head_speed = choose_side(head_speed_left, head_speed_right, side)
		tail_speed = choose_side(tail_speed_left, tail_speed_right, side)

		#this is used to flip the direction of the inequallity in the branching between the left and the right side
		#the xor which will be in the following lines uses this boolean like some "controlled not"
		#when right_condition is "true" the inequallity will be flipped
		#when right_condition is "false" the inequallity will be stay the same
		right_condition = isa(side, RightSide)

		if xor(S < head_speed, right_condition)
			profile[i] = choose_side(status_left, status_right, side)
		elseif xor(S < tail_speed, right_condition) #can only happen in Rarefaction, because in Shock  head_speed == tail_speed
			profile[i] = rarefaction_profile(S, status, side)
		else
			profile[i] = choose_side(status_left_star, status_right_star, side)
		end
	end

	return profile
end

#sample the riemann problem in the situation where the gas in "side" is actually a vaccum
function sample_riemann_side_vaccum(x, t::Float64, status::HydroStatus, side::T) where {T<:Side}

	profile = similar(x, HydroStatusSimple) #we return the values of rho, u and p in each point x in space
	vaccum = HydroStatusSimple(0., 0., 0.) #vaccum
	simple = HydroStatusSimple(status)

	x_over_t = x ./ t

	tail_speed = tail_speed_calc(0., 0., status, side, Vaccum()) #the head of the rarefaciton wave
	head_speed = head_speed_calc(0., 0., status, side, Vaccum()) #the tail of the rarefaciton wave (=the boundary with the vaccum)

	#see "sample_riemann_regular" for explnation about this boolean
	right_condition = isa(side, RightSide)

	for i = 1:length(x)
		S = x_over_t[i]

		if xor(S < head_speed, right_condition)
			profile[i] = simple
		elseif xor(S < tail_speed, right_condition)
			profile[i] = rarefaction_profile(S, status, side)
		else
			profile[i] = vaccum
		end
	end

	return profile
end
#
function RiemannSolver(x, t::Float64,left_rho::Float64,left_u::Float64,left_v::Float64,left_p::Float64,left_gamma::Float64,
									 right_rho::Float64,right_u::Float64,right_v::Float64,right_p::Float64,right_gamma::Float64;
									 TOL::Float64 = 1.e-6)
	left=HydroStatus(left_rho,left_u,left_p,left_gamma)
	right=HydroStatus(right_rho,right_u,right_p,right_gamma)
	profile=RiemannSolver(x, t,left,right;TOL=TOL)

	# so far can only handel the case no vacuum appear:
	@assert (left.rho>0) && (right.rho>0) && (2. / (left.gamma - 1.) * (left.c + right.c) > right.u - left.u)
	p_star = p_star_calc(left, right, TOL)
	u_star = u_star_calc(p_star, left, right)
	x_over_t = x ./ t
	profile_v=similar(x,Float64)
	for i = 1:length(x)
		S = x_over_t[i] #this is like the S which Toro use in Section 4.5
		if S>u_star
			profile_v[i]=right_v
		else
			profile_v[i]=left_v
		end
	end
	return profile, profile_v
end
