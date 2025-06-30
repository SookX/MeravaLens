import { useContext, useEffect, useRef, useState } from "react"
import AccountForm from "../../components/AccountForm/AccountForm"
import { DataContext } from "../../context/DataContext"
import FormPage from "../../components/FormPage/FormPage"
import { GoogleLogin } from '@react-oauth/google'
import { Box, Stack } from "@mui/material"
import GoogleButton from "../../components/GoogleButton/GoogleButton"

const Login = () => {
    // Gets global data from the context
    const { crud, access, setAccess, setRefresh, navigate } = useContext(DataContext)



    // Checks if the user is already authenticated
    useEffect(() => {
        if(access) navigate('/dashboard')
    }, [access])



    // Holds the values for the form
    const emailRef = useRef()
    const passwordRef = useRef()
    const [error, setError] = useState(null)

    const inputs = [
        {
            type: "email",
            label: "Email",
            ref: emailRef
        },
        {
            type: "password",
            label: "Password",
            ref: passwordRef
        }
    ]



    // Makes a login request to the backend
    const handleSubmit = async () => {
        const response = await crud({
            url: "/users/login/",
            method: "post",
            body: {
                email: emailRef.current.value,
                password: passwordRef.current.value,
            }
        })

        if(response.status == 200) {
            localStorage.setItem('access', response.data.token.access)
            setAccess(response.data.token.access)
            localStorage.setItem('refresh', response.data.token.refresh)
            setRefresh(response.data.token.refresh)
            navigate('/dashboard')
        }
        else setError(response.response.data.error)

        console.log(response)
    }



    return (
        <FormPage>
            <AccountForm
                title="Welcome back!"
                text="Enter your credentials and get back to the exciting world of satellites."
                error={error}
                inputs={inputs}
                forgotPassword={true}
                handleSubmit={handleSubmit}
                link={{
                    link: "/register",
                    text: "You don't have an account?",
                    label: "Sign up"
                }}
                buttonLabel="Log in to my account"
                oauth={{
                    component: (
                        <GoogleButton
                            setError={setError}
                            text="signin_with"
                        />
                    )
                }}
            />
        </FormPage>
    )
}

export default Login