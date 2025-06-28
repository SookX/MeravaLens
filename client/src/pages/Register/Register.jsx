import { useContext, useState } from "react"
import AccountForm from "../../components/AccountForm/AccountForm"
import { DataContext } from "../../context/DataContext"
import { Email, Person, LockOutline } from '@mui/icons-material'

const Register = () => {
    // Gets global data from the context
    const { crud } = useContext(DataContext)

    const [email, setEmail] = useState("")
    const [username, setUsername] = useState("")
    const [password, setPassword] = useState("")
    const [confirmPassword, setConfirmPassword] = useState("")

    const inputs = [
        {
            type: "email",
            label: "Email",
            value: email,
            setValue: setEmail,
            // icon: (<Email />)
        },
        {
            type: "text",
            label: "Username",
            value: username,
            setValue: setUsername,
            // icon: (<Person />)
        },
        {
            type: "password",
            label: "Password",
            value: password,
            setValue: setPassword,
            // icon: (<LockOutline />)
        },
        {
            type: "password",
            label: "Confirm Password",
            value: confirmPassword,
            setValue: setConfirmPassword,
            // icon: (<LockOutline />)
        },
    ]

    const handleSubmit = async () => {
        const response = await crud({
            url: "/register",
            method: "post",
            body: {
                email,
                username,
                password,
                confirmPassword
            }
        })

        console.log(response)
    }

    return (
        <AccountForm
            title="Make an account."
            text="Make a Merava Lens account to get the latest satellite analysis."
            inputs={inputs}
            handleSubmit={handleSubmit}
            link={{
                link: "/login",
                text: "Alredy have an account?",
                label: "Log in"
            }}
        />
    )
}

export default Register